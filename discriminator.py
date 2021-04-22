import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, args, src_dict, dst_dict, emb_dim=50, use_cuda=True, dropout=0.1, num_heads=1, layers=1):
        super(AttDiscriminator, self).__init__()
        vocab_size = 200000 # len(src_dict)
        self.embed_src_tokens = self.embed_trg_tokens = nn.Embedding(vocab_size, emb_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(emb_dim, num_heads, dim_feedforward=emb_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layers)
        self.mask = self.generate_square_subsequent_mask(1)

        self.fc = nn.Linear(emb_dim, 1)

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, source_ids, target_ids):
        return checkpoint.checkpoint(self.do_stuff, source_ids, target_ids, self.dummy_tensor)

    def do_stuff(self, source_ids, target_ids, dummy=None):
        source_emb = self.embed_src_tokens(source_ids).permute(1, 0, 2)
        target_emb = self.embed_trg_tokens(target_ids).permute(1, 0, 2)
        if self.mask.size(0) != target_emb.size(0):
            self.mask = self.generate_square_subsequent_mask(target_emb.size(0)).to(source_ids.device)
        out = self.decoder(target_emb, source_emb, tgt_mask=self.mask)

        out = self.fc(out)

        return torch.tanh(out.permute(1, 0, 2).squeeze(2))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class GumbelDiscriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, emb_dim=50, use_cuda=True, dropout=0.1, num_heads=1, layers=1):
        super(GumbelDiscriminator, self).__init__()
        vocab_size = 200000 # len(src_dict)
        self.embed_src_tokens = self.embed_trg_tokens = nn.Embedding(vocab_size, emb_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(emb_dim, num_heads, dim_feedforward=emb_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layers)
        self.mask = self.generate_square_subsequent_mask(1)

        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, source_onehot, target_onehot):
        source_emb = (source_onehot @ self.embed_src_tokens.weight[:source_onehot.shape[-1], :]).permute(1, 0, 2)
        target_emb = (target_onehot @ self.embed_trg_tokens.weight[:target_onehot.shape[-1], :]).permute(1, 0, 2)
        if self.mask.size(0) != target_emb.size(0):
            self.mask = self.generate_square_subsequent_mask(target_emb.size(0)).to(source_onehot.device)
        out = self.decoder(target_emb, source_emb, tgt_mask=self.mask)

        out = self.fc(out)

        return torch.tanh(out.permute(1, 0, 2).squeeze(2))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


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