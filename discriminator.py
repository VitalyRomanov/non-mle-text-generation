import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from SeqT5 import SeqT5_Discriminator


# class AttDiscriminator(nn.Module):
#     def __init__(self, args, src_dict, dst_dict, use_cuda=True, dropout=0.1, num_heads=1):
#         super(AttDiscriminator, self).__init__()
#
#         # TODO resolve the problem
#         self.src_dict_size = 200000 #len(src_dict)
#         self.trg_dict_size = 200000 #len(dst_dict)
#
#         self.src_pad_idx = src_dict.pad()
#         self.pad_idx = dst_dict.pad()
#         self.fixed_max_len = args.fixed_max_len
#         self.use_cuda = use_cuda
#
#         assert args.encoder_embed_dim == args.decoder_embed_dim
#
#         emb_dim = args.decoder_out_embed_dim
#
#         # TODO share this across encoder and decoder
#         self.embed_src_tokens = Embedding(self.src_dict_size, emb_dim, src_dict.pad())
#         self.embed_trg_tokens = Embedding(self.trg_dict_size, emb_dim, dst_dict.pad())
#
#         self.attention = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout)
#         self.input_proj = Linear(emb_dim * num_heads, 1, bias=False)
#         self.prediction = Linear(emb_dim * num_heads, 1, bias=False)
#
#     def forward(self, src_sentence, trg_sentence):
#         src_out = self.embed_src_tokens(src_sentence)
#         trg_out = self.embed_trg_tokens(trg_sentence)
#
#         src_mask = src_sentence == self.src_pad_idx
#         trg_mask = trg_sentence == self.pad_idx
#
#         input = torch.cat([src_out, trg_out], dim=1)
#         input_mask = torch.cat([src_mask, trg_mask], dim=1)
#
#         query = key = value = input.permute(1, 0, 2)
#
#         mh_att, _ = self.attention(query, key, value, key_padding_mask=input_mask)
#         mh_att = mh_att.permute(1, 0, 2)
#
#         attn_logits = self.input_proj(mh_att)
#         attn_logits = attn_logits.squeeze(2)
#         attn_scores = F.softmax(attn_logits, dim=1).unsqueeze(2)
#
#         x = (attn_scores * mh_att).sum(dim=1)
#
#         logits =  self.prediction(x)
#         return torch.sigmoid(logits)


# class TokenDiscriminator(nn.Module):
#     # TODO replace with LM
#     def __init__(self, args, src_dict, dst_dict, use_cuda=True, dropout=0.1, num_heads=1):
#         super().__init__()
#
#         # TODO resolve the problem
#         self.src_dict_size = 200000 #len(src_dict)
#         self.trg_dict_size = 200000 #len(dst_dict)
#
#         self.src_pad_idx = src_dict.pad()
#         self.pad_idx = dst_dict.pad()
#         self.fixed_max_len = args.fixed_max_len
#         self.use_cuda = use_cuda
#         self.num_heads = num_heads
#
#         assert args.encoder_embed_dim == args.decoder_embed_dim
#
#         emb_dim = 30
#
#         # TODO share this across encoder and decoder
#         self.embed_src_tokens = Embedding(self.src_dict_size, emb_dim, src_dict.pad())
#         self.embed_trg_tokens = Embedding(self.trg_dict_size, emb_dim, dst_dict.pad())
#
#         self.attention = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout)
#         self.attention2 = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout)
#         self.prediction = Linear(emb_dim * num_heads, 1, bias=False)
#
#     def forward(self, src_sentence, trg_sentence):
#         src_out = self.embed_src_tokens(src_sentence)
#         trg_out = self.embed_trg_tokens(trg_sentence)
#
#         src_mask = ~(src_sentence == self.src_pad_idx)
#         # trg_mask = ~(trg_sentence == self.pad_idx)
#         # attn_mask = (src_mask.unsqueeze(2) * trg_mask.unsqueeze(1)).repeat(self.num_heads, 1, 1)
#
#         query = trg_out.permute(1, 0, 2)
#         key = value = src_out.permute(1, 0, 2)
#
#         mh_att, _ = self.attention(query, key, value)
#         mh_att, _ = self.attention2(mh_att, mh_att, mh_att) #, key_padding_mask=trg_mask)
#
#         mh_att = mh_att.permute(1, 0, 2)
#
#         logits =  self.prediction(mh_att).squeeze(2)
#         return torch.sigmoid(logits)
from torch.utils import checkpoint


class AttDiscriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, emb_dim=50, use_cuda=True, dropout=0.1, num_heads=1, layers=3):
        super(AttDiscriminator, self).__init__()
        vocab_size = 200000 # len(src_dict)
        self.create_embedders(vocab_size, emb_dim)
        # self.decoder_layer = nn.TransformerDecoderLayer(emb_dim, num_heads, dim_feedforward=emb_dim)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layers)
        self.encoder_layer = nn.TransformerEncoderLayer(emb_dim, num_heads, dim_feedforward=emb_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        self.target_mask = self.generate_square_subsequent_mask(1)
        # self.memory_mask = self.generate_square_subsequent_mask(1)

        self.fc = nn.Linear(emb_dim, 1)

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def create_embedders(self, vocab_size, emb_dim):
        self.embed_src_tokens = self.embed_trg_tokens = nn.Embedding(vocab_size, emb_dim)

    def forward(self, source_ids, target_ids):
        return checkpoint.checkpoint(self.do_stuff, source_ids, target_ids, self.dummy_tensor)

    def get_tgt_embeddings(self, tokens):
        return self.embed_trg_tokens(tokens).permute(1, 0, 2)

    def do_stuff(self, source_ids, target_ids, dummy=None):
        # source_emb = self.embed_src_tokens(source_ids).permute(1, 0, 2)
        # target_emb = self.embed_trg_tokens(target_ids).permute(1, 0, 2)
        target_emb = self.get_tgt_embeddings(target_ids)
        if self.target_mask.size(0) != target_emb.size(0):
            self.target_mask = self.generate_square_subsequent_mask(target_emb.size(0)).to(source_ids.device)
        # if self.memory_mask.size(0) != source_emb.size(0):
        #     self.memory_mask = self.generate_square_subsequent_mask(source_emb.size(0)).to(source_ids.device)
        # out = self.decoder(target_emb, source_emb, tgt_mask=self.target_mask, memory_mask=self.target_mask)
        target_emb = self.pos_encoder(target_emb)
        out = self.encoder(target_emb, mask=self.target_mask)

        out = self.fc(out)

        return torch.sigmoid(out.permute(1, 0, 2).squeeze(2))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class T5Discriminator(AttDiscriminator):
    def __init__(self, args, src_dict, dst_dict, emb_dim=512, use_cuda=True, dropout=0.1, num_heads=1, layers=1):
        super(T5Discriminator, self).__init__(args, src_dict, dst_dict, emb_dim=emb_dim, use_cuda=use_cuda, dropout=dropout, num_heads=num_heads, layers=layers)

    def create_embedders(self, vocab_size, emb_dim):
        self.t5_model = SeqT5_Discriminator.from_pretrained('t5-small')

    def get_tgt_embeddings(self, tokens):
        return self.t5_model(tokens - 1, return_encoder=True).permute(1, 0, 2)


class T5SemanticDiscriminator(nn.Module):
    def __init__(self):
        super(T5SemanticDiscriminator, self).__init__()
        self.t5_model = SeqT5_Discriminator.from_pretrained('t5-small')
        # self.encoder_layer = nn.TransformerEncoderLayer(512, 1, dim_feedforward=512)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.pos_encoder = PositionalEncoding(512, 0.1)
        # self.fc = nn.Linear(512, 1)
        #
        # self.target_mask = self.generate_square_subsequent_mask(1)

    def transform_for_t5(self, tensor):
        return tensor - 1

    def forward(self, source_ids, target_ids):
        out = self.t5_model(
            self.transform_for_t5(source_ids), labels=self.transform_for_t5(target_ids), return_encoder=False
        )
        # if self.target_mask.size(0) != target_emb.size(0):
        #     self.target_mask = self.generate_square_subsequent_mask(target_emb.size(0)).to(source_ids.device)

        # target_emb = self.pos_encoder(target_emb)
        # out = self.encoder(target_emb, mask=self.target_mask)
        # out = self.fc(out)
        return torch.sigmoid(out.squeeze(2))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GumbelDiscriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, emb_dim=50, use_cuda=True, dropout=0.1, num_heads=1, layers=1):
        super(GumbelDiscriminator, self).__init__()
        vocab_size = 200000 # len(src_dict)
        self.embed_src_tokens = self.embed_trg_tokens = nn.Embedding(vocab_size, emb_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(emb_dim, num_heads, dim_feedforward=emb_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layers)
        self.mask = self.generate_square_subsequent_mask(1)
        # self.emb_w = nn.Parameter(torch.rand(vocab_size, emb_dim), requires_grad=True)

        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, source_onehot, target_onehot):
        source_emb = (source_onehot @ self.embed_src_tokens.weight[:source_onehot.shape[-1], :]).permute(1, 0, 2)
        target_emb = (target_onehot @ self.embed_trg_tokens.weight[:target_onehot.shape[-1], :]).permute(1, 0, 2)
        return self.do_stuff(source_emb, target_emb)# checkpoint.checkpoint(self.do_stuff, source_emb, target_emb)

    def do_stuff(self, source_emb, target_emb):

        if self.mask.size(0) != target_emb.size(0):
            self.mask = self.generate_square_subsequent_mask(target_emb.size(0)).to(source_emb.device)
        out = self.decoder(target_emb, source_emb, tgt_mask=self.mask)

        out = self.fc(out)

        return torch.sigmoid(out.permute(1, 0, 2).squeeze(2))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class BleurtDiscriminator(nn.Module):
    def __init__(self, decode_fn):
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
        from bleurt import score as bleurt_score
        super(BleurtDiscriminator, self).__init__()
        checkpoint = "bleurt/bleurt-base-128"
        self.scorer = bleurt_score.BleurtScorer(checkpoint)
        self.decode_fn = decode_fn

    def forward(self, prediction, target):
        candidates, references = self.decode_fn(prediction), self.decode_fn(target)
        scores = self.scorer.score(references=references, candidates=candidates)
        return torch.Tensor(scores).reshape(-1,1)


class Discriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda = True):
        super(Discriminator, self).__init__()

        self.src_dict_size = len(src_dict)
        self.trg_dict_size = len(dst_dict)
        self.pad_idx = dst_dict.pad()
        self.fixed_max_len = args.fixed_max_len
        self.use_cuda = use_cuda

        self.embed_src_tokens = Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
        self.embed_trg_tokens = Embedding(len(dst_dict), args.decoder_embed_dim, dst_dict.pad())


        self.conv1 = nn.Sequential(
            Conv2d(in_channels=2000,
                   out_channels=512,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=512,
                   out_channels=256,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 12 * 12, 20),
            nn.ReLU(),
            nn.Dropout(),
            Linear(20, 20),
            nn.ReLU(),
            Linear(20, 1),
        )

    def forward(self, src_sentence, trg_sentence):
        batch_size = src_sentence.size(0)

        src_out = self.embed_src_tokens(src_sentence)
        trg_out = self.embed_src_tokens(trg_sentence)

        src_out = torch.stack([src_out] * trg_out.size(1), dim=2)
        trg_out = torch.stack([trg_out] * src_out.size(1), dim=1)
        
        out = torch.cat([src_out, trg_out], dim=3)
        
        out = out.permute(0,3,1,2)
        
        out = self.conv1(out)
        out = self.conv2(out)
        
        out = out.permute(0, 2, 3, 1)
        
        out = out.contiguous().view(batch_size, -1)
        
        out = torch.sigmoid(self.classifier(out))

        return out

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        nn.init.constant_(m.bias.data, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m