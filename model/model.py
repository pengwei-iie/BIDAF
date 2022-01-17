import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)    # 遇到padding_idx=1的，用0填充
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width))     # 100， 8， 5

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe # the tensor does not get updated in the learning process.
        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)

        # highway network
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = nn.LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = nn.Linear(args.hidden_size * 2, 1)
        self.att_weight_q = nn.Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = nn.Linear(args.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = nn.LSTM(input_size=args.hidden_size * 8,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        self.modeling_LSTM2 = nn.LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        # 6. Output Layer
        self.p1_weight_g = nn.Linear(args.hidden_size * 8, 1)
        self.p1_weight_m = nn.Linear(args.hidden_size * 2, 1)
        self.p1_linear = nn.Linear(2, 1)
        self.p2_weight_g = nn.Linear(args.hidden_size * 8, 1)
        self.p2_weight_m = nn.Linear(args.hidden_size * 2, 1)
        self.p2_linear = nn.Linear(2, 1)

        self.output_LSTM = nn.LSTM(input_size=args.hidden_size * 2,
                                hidden_size=args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        # @torchsnooper.snoop()
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)  # self.highway_linear0(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)    # self.highway_gate0(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        # @torchsnooper.snoop()
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, hidden*8)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            s = torch.bmm(c, q.reshape(q.size()[0], q.size()[-1], -1))
            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            # p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze(-1)
            p1 = self.p1_linear(torch.cat([self.p1_weight_g(g), self.p1_weight_m(m)], dim=-1)).squeeze(-1)
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM(m)[0]
            # (batch, c_len)
            p2 = self.p2_linear(torch.cat([self.p2_weight_g(g), self.p2_weight_m(m2)], dim=-1)).squeeze(-1)

            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]                    #
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM(c)[0]
        q = self.context_LSTM(q)[0]       #
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2(self.modeling_LSTM1(g)[0])[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2
