import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils


class LSTMModel(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda=True):
        super(LSTMModel, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        self.src_dict = src_dict
        self.dst_dict = dst_dict

        # Initialize encoder and decoder
        self.create_encoder(args)
        self.create_decoder(args)

    def create_encoder(self, args):
        self.encoder = LSTMEncoder(
            self.src_dict,
            embed_dim=args.encoder_embed_dim,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
        )

    def create_decoder(self, args):
        self.decoder = LSTMDecoder(
            self.dst_dict,
            encoder_embed_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            use_cuda=self.use_cuda
        )

    def forward(self, sample, inference=False):
        # encoder_output: (seq_len, batch, hidden_size * num_directions)
        # _encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        # _encoder_cell: (num_layers * num_directions, batch, hidden_size)
        encoder_out = self.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])  # TODO what is net input
        
        # # The encoder hidden is  (layers*directions) x batch x dim.   
        # # If it's bidirectional, We need to convert it to layers x batch x (directions*dim).
        # if self.args.bidirectional:
        #     encoder_hiddens = torch.cat([encoder_hiddens[0:encoder_hiddens.size(0):2], encoder_hiddens[1:encoder_hiddens.size(0):2]], 2)
        #     encoder_cells = torch.cat([encoder_cells[0:encoder_cells.size(0):2], encoder_cells[1:encoder_cells.size(0):2]], 2)

        decoder_out, attn_scores = self.decoder(sample['net_input']['prev_output_tokens'], encoder_out, inference=inference)
        decoder_out = F.log_softmax(decoder_out, dim=2)
        
        # sys_out_batch = decoder_out.contiguous().view(-1, decoder_out.size(-1))
        # loss = F.nll_loss(sys_out_batch, train_trg_batch, reduction='sum', ignore_index=self.dst_dict.pad())        
        
        return decoder_out

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        vocab = net_output.size(-1)
        net_output1 = net_output.view(-1, vocab)
        if log_probs:
            return F.log_softmax(net_output1, dim=1).view_as(net_output)
        else:
            return F.softmax(net_output1, dim=1).view_as(net_output)


class VarLSTMModel(LSTMModel):
    def __init__(self, args, src_dict, dst_dict, use_cuda=True):
        super(VarLSTMModel, self).__init__(args, src_dict, dst_dict, use_cuda=use_cuda)

    def create_decoder(self, args):
        self.decoder = VarLSTMDecoder(
            self.dst_dict,
            encoder_embed_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            use_cuda=self.use_cuda
        )

    def forward(self, sample, inference=False):
        encoder_out = self.encoder(sample['net_input']['src_tokens'],
                                   sample['net_input']['src_lengths'])  # TODO what is net input

        decoder_out, attn_scores, kld = self.decoder(sample['net_input']['prev_output_tokens'], encoder_out, inference=inference)
        decoder_out = F.log_softmax(decoder_out, dim=2)

        return decoder_out, kld


class LSTMEncoder(nn.Module):
    """LSTM encoder."""
    def __init__(self, dictionary, embed_dim=512, num_layers=1, dropout_in=0.1, dropout_out=0.1):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=self.dropout_out,
            bidirectional=False,
        )

    def forward(self, src_tokens, src_lengths):

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # # pack embedded source tokens into a PackedSequence
        # packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        h0 = Variable(x.data.new(self.num_layers, bsz, embed_dim).zero_())
        c0 = Variable(x.data.new(self.num_layers, bsz, embed_dim).zero_())
        x, (final_hiddens, final_cells) = self.lstm(
            x,
            (h0, c0),
        )

        # unpack outputs and apply dropout
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, embed_dim]

        return x, final_hiddens, final_cells

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2*output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=1).t()  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(nn.Module):
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True):
        super(LSTMDecoder, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.create_layers(encoder_embed_dim, embed_dim, num_layers)

        self.attention = AttentionLayer(encoder_embed_dim, embed_dim)
        if embed_dim != out_embed_dim:
            self.additional_fc = Linear(embed_dim, out_embed_dim)
        self.pre_fc1 = Linear(out_embed_dim, out_embed_dim, dropout=dropout_out)
        self.pre_fc2 = Linear(out_embed_dim, out_embed_dim, dropout=dropout_out)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def create_layers(self, encoder_embed_dim, embed_dim, num_layers):
        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else embed_dim, embed_dim)
            for layer in range(num_layers)
        ])

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, inference=False):
        if incremental_state is not None:  # TODO what is this?
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens) # (bze, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)

        # initialize previous states (or get from cache during incremental generation)
        # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = Variable(x.data.new(bsz, embed_dim).zero_())

        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                input = input / torch.norm(input, dim=-1).unsqueeze(1)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        x = nn.functional.relu(self.pre_fc1(x))
        x = nn.functional.relu(self.pre_fc2(x))
        x = self.fc_out(x)

        return x, attn_scores


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def reorder_incremental_state(self, incremental_state, new_order):
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        if not isinstance(new_order, Variable):
            new_order = Variable(new_order)
        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)


class LSTMEmbDecoder(LSTMDecoder):
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True, pretrained_embeddings=None):
        super(LSTMEmbDecoder, self).__init__(
            dictionary, encoder_embed_dim, embed_dim, out_embed_dim, num_layers, dropout_in, dropout_out, use_cuda
        )
        self.fc_out = Linear(out_embed_dim, embed_dim, dropout=dropout_out)
        if pretrained_embeddings is not None:
            import numpy as np
            pretrained_embeddings = torch.from_numpy(np.load(pretrained_embeddings)).float()
            if pretrained_embeddings.shape[0] > self.embed_tokens.weight.shape[0]:
                pretrained_embeddings = pretrained_embeddings[:self.embed_tokens.weight.shape[0], :]
            assert self.embed_tokens.weight.shape == pretrained_embeddings.shape
            norm = torch.norm(pretrained_embeddings, dim=-1).unsqueeze(1)
            self.embed_tokens.weight = torch.nn.Parameter(pretrained_embeddings / norm)
            self.embed_tokens.weight.requires_grad = False

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, inference=False):
        if incremental_state is not None:  # TODO what is this?
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens) # (bze, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)

        # initialize previous states (or get from cache during incremental generation)
        # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = Variable(x.data.new(bsz, embed_dim).zero_())

        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                input = input / torch.norm(input, dim=-1).unsqueeze(1)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        x = nn.functional.relu(self.pre_fc1(x))
        x = nn.functional.relu(self.pre_fc2(x))
        x = self.fc_out(x)
        x = x / torch.norm(x, dim=-1).unsqueeze(-1)

        return x, attn_scores


class LSTMEmbHelperDecoder(LSTMDecoder):
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True, pretrained_embeddings=None):
        super(LSTMEmbHelperDecoder, self).__init__(
            dictionary, encoder_embed_dim, embed_dim, out_embed_dim, num_layers, dropout_in, dropout_out, use_cuda
        )
        self.fc_out = Linear(out_embed_dim, embed_dim, dropout=dropout_out)
        if pretrained_embeddings is not None:
            import numpy as np
            pretrained_embeddings = torch.from_numpy(np.load(pretrained_embeddings)).float()
            if pretrained_embeddings.shape[0] > self.embed_tokens.weight.shape[0]:
                pretrained_embeddings = pretrained_embeddings[:self.embed_tokens.weight.shape[0], :]
            assert self.embed_tokens.weight.shape == pretrained_embeddings.shape
            norm = torch.norm(pretrained_embeddings, dim=-1).unsqueeze(1)
            self.embed_tokens.weight = torch.nn.Parameter(pretrained_embeddings / norm)
            self.embed_tokens.weight.requires_grad = False


class LSTMEmbModel(LSTMModel):
    def __init__(self, *args, **kwargs):
        super(LSTMEmbModel, self).__init__(*args, **kwargs)

    def create_decoder(self, args):
        self.decoder = LSTMEmbDecoder(
            self.dst_dict,
            encoder_embed_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            use_cuda=self.use_cuda,
            pretrained_embeddings = args.pretrained_embeddings
        )

    def forward(self, sample, inference=False):
        encoder_out = self.encoder(sample['net_input']['src_tokens'],
                                   sample['net_input']['src_lengths'])  # TODO what is net input

        decoder_out, attn_scores = self.decoder(sample['net_input']['prev_output_tokens'], encoder_out,
                                                inference=inference)
        return decoder_out


class VarLSTMDecoder(LSTMDecoder):
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True):
        super(VarLSTMDecoder, self).__init__(
            dictionary, encoder_embed_dim=encoder_embed_dim, embed_dim=embed_dim,
            out_embed_dim=out_embed_dim, num_layers=num_layers, dropout_in=dropout_in,
            dropout_out=dropout_out, use_cuda=use_cuda
        )

        self.context_to_mu = nn.Linear(embed_dim, embed_dim)
        self.context_to_logvar = nn.Linear(embed_dim, embed_dim)

    def create_layers(self, encoder_embed_dim, embed_dim, num_layers):
        # encoder_embed_dim * 2 for latent code
        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim * 2 + embed_dim if layer == 0 else embed_dim, embed_dim)
            for layer in range(num_layers)
        ])

    def modify_state(self, hidden_state):
        return self.reparameterize(hidden_state)

    def reparameterize(self, hidden):
        """
        context [B x 2H]
        """
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z, mu, logvar

    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, inference=False):
        if incremental_state is not None:  # TODO what is this?
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens) # (bze, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)

        # initialize previous states (or get from cache during incremental generation)
        # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = Variable(x.data.new(bsz, embed_dim).zero_())

        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())
        outs = []

        kld = 0.
        z, mu, logvar = self.reparameterize(prev_hiddens[0])
        kld += self.compute_kld(mu, logvar)

        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            if inference is True:
                input = torch.cat([input, mu], dim=1)
            else:
                input = torch.cat([input, z], dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        x = self.fc_out(x)

        if inference is True:
            return x, attn_scores
        else:
            return x, attn_scores, kld


class TokenRecovery(nn.Module):
    def __init__(self, input_size, output_size):
        super(TokenRecovery, self).__init__()
        self.extra_token_clf_layer1 = torch.nn.Linear(input_size, input_size)
        self.extra_token_clf_layer2 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.extra_token_clf_layer1(x)
        x = torch.relu(x)
        x = self.extra_token_clf_layer2(x)
        return x


# TODO why they use this specific initialization
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m