import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class mGRU(nn.Module):
    """matchLSTM implementation but using GRU instead of LSTM
    """
    def __init__ (self, options):
        super(mGRU, self).__init__()
        self.options = options
        self.n_embed = self.options["EMBED_DIM"]
        self.n_dim = self.options["HIDDEN_DIM"]
        self.n_out = self.options["CLASSES"]
        device = torch.device("cuda:0" if self.options["CUDA"] else "cpu")

        self.compress = nn.Linear(768, self.n_embed).to(device)
        self.premise_gru = nn.GRU(self.n_embed, self.n_dim, bidirectional=False).to(device)
        self.hypothesis_gru = nn.GRU(self.n_embed, self.n_dim, bidirectional=False).to(device)
        self.out = nn.Linear(self.n_dim, self.n_out).to(device)
        
        # Attention Parameters
        if self.options["CUDA"]:
            self.W_s = nn.Parameter(torch.randn(self.n_dim, self.n_dim).to(device))  # n_dim x n_dim
            self.register_parameter('W_s', self.W_s)
            self.W_t = nn.Parameter(torch.randn(self.n_dim, self.n_dim).to(device))  # n_dim x n_dim
            self.register_parameter('W_t', self.W_t)
            self.w_e = nn.Parameter(torch.randn(self.n_dim, 1).to(device)) # n_dim x 1
            self.register_parameter('w_e', self.w_e)
            self.W_m = nn.Parameter(torch.randn(self.n_dim, self.n_dim).to(device))  # n_dim x n_dim
            self.register_parameter('W_m', self.W_m)
        else:
            self.W_s = nn.Parameter(torch.randn(self.n_dim, self.n_dim))
            self.register_parameter('W_s', self.W_s)
            self.W_t = nn.Parameter(torch.randn(self.n_dim, self.n_dim))
            self.register_parameter('W_t', self.W_t)
            self.w_e = nn.Parameter(torch.randn(self.n_dim, 1))
            self.register_parameter('w_e', self.w_e)
            self.W_m = nn.Parameter(torch.randn(self.n_dim, self.n_dim))
            self.register_parameter('W_m', self.W_m)

        # Match GRU parameters.
        self.m_gru = nn.GRU(self.n_dim + self.n_dim, self.n_dim, bidirectional=False).to(device)

    def _init_hidden(self, batch_size):
        """Init hidden matrix for GRU"""
        hidden_p = Variable(torch.zeros(1, batch_size, self.n_dim))
        hidden_h = Variable(torch.zeros(1, batch_size, self.n_dim))
        return hidden_p, hidden_h

    def _attn_gru_init_hidden(self, batch_size):
        """Init for GRU attention"""
        r_0 = Variable(torch.zeros(batch_size, self.n_dim))
        return r_0

    def mask_mult(self, o_t, o_tm1, mask_t):
        """"""
        return (o_t.to(device) * mask_t.to(device)) + (o_tm1.to(device) * (torch.logical_not(mask_t.to(device))))
    
    def _gru_forward(self, gru, encoded_sent, mask_sent, h_0):
        """Stateful GRU for premise/hypothesis

        Parameters:
        ----
         - gru: GRU cell
         - encoded_sent: embedded matrix of premise/hypothesis sentence
         - mask_sent: mask vector for embedded matrix
         - h_0: init hidden vector for GRU cell

        Returns:
        ----
         - o_s: output of last timestep in each batch from GRU cell. A matrix has shape (seq_len x batch x n_dim)
         - h_t: last hidden state vector (1 x batch x n_dim)
        """
        len_seq = encoded_sent.size(0)
        batch_size = encoded_sent.size(1)
        o_s = Variable(torch.zeros(len_seq, batch_size, self.n_dim))
        h_tm1 = h_0.squeeze(0)
        o_tm1 = None

        for ix, (x_t, mask_t) in enumerate(zip(encoded_sent, mask_sent)):
            '''
            x_t : batch x n_embed; 
            mask_t : batch,
            '''
            o_t, h_t = gru(x_t.unsqueeze(0).to(device), 
                           h_tm1.unsqueeze(0).to(device))  # 1 x batch x n_dim
            mask_t = mask_t.unsqueeze(1)  # batch x 1
            h_t = self.mask_mult(h_t[0], h_tm1, mask_t)

            if o_tm1 is not None:
                o_t = self.mask_mult(o_t[0], o_tm1, mask_t)
            o_tm1 = o_t[0] if o_tm1 is None else o_t
            h_tm1 = h_t
            o_s[ix] = o_t

        return o_s, h_t.unsqueeze(0)

    def _attention_forward(self, H_s, mask_H_s, h_t, h_m_tm1=None):
        '''Word-by-word attention. https://arxiv.org/pdf/1509.06664.pdf

        Computes the Attention Weights over H_s using h_t (and h_m_tm1 if given)
        Returns an attention weighted representation of H_s, and the alphas.

        Parameters:
        ----
            H_s (seq_len x batch x n_dim): output of all batchs come from GRU cell
            mask_Y (seq_len x batch): mask matrix
            h_t (batch x n_dim): hidden matrix for t-th word in hypothesis (batch)
            h_m_tm1 (batch x n_dim): previous h_m

        Returns:
        ----
            h_m (batch x n_dim)
            alpha (batch x T): attention weight
        '''
        H_s = H_s.transpose(1, 0).cuda()  # batch x seq_len x n_dim
        mask_H_s = mask_H_s.transpose(1, 0)  # batch x seq_len

        Whs = torch.bmm(H_s, self.W_s.unsqueeze(0).expand(H_s.size(0), *self.W_s.size()))  # batch x seq_len x n_dim
        Wht = torch.mm(h_t.cuda(), self.W_t)  # batch x n_dim
        if h_m_tm1 is not None:
            W_r_tm1 = torch.mm(h_m_tm1.cuda(), self.W_m)  # (batch, n_dim)
            Whs += W_r_tm1.unsqueeze(1)
        M = torch.tanh(Whs + Wht.unsqueeze(1).expand(Wht.size(0), H_s.size(1), Wht.size(1)))  # batch x seq_len x n_dim
        alpha = torch.bmm(M, self.w_e.unsqueeze(0).expand(H_s.size(0), *self.w_e.size())).squeeze(-1)  # batch x seq_len
        alpha = alpha + (-1000.0 * (torch.logical_not(mask_H_s)))  # To ensure probability mass doesn't fall on non tokens
        alpha = F.softmax(alpha)
        return torch.bmm(alpha.unsqueeze(1), H_s).squeeze(1), alpha

    def _attn_gru_forward(self, o_h, mask_h, r_0, o_p, mask_p):
        '''Use match-GRU to modeling the matching between the premise and the hypothesis.

        Parameters:
        ----
         - o_h : seq_len x batch x n_dim : The hypothesis
         - mask_h : seq_len x batch
         - r_0 : batch x n_dim :
         - o_p : seq_len x batch x n_dim : The premise. Will attend on it at every step
         - mask_p : seq_len x batch : the mask for the premise

        Returns:
        ----
         - r : batch x n_dim : the last state of the RNN
         - alpha_vec : seq_len x batch x seq_len the attn vec at every step
        '''
        seq_len_h = o_h.size(0)
        batch_size = o_h.size(1)
        seq_len_p = o_p.size(0)
        alpha_vec = Variable(torch.zeros(seq_len_h, batch_size, seq_len_p))
        r_tm1 = r_0
        for ix, (h_t, mask_t) in enumerate(zip(o_h, mask_h)):
            '''
                h_t : batch x n_dim
                mask_t : batch,
            '''
            a_t, alpha = self._attention_forward(o_p, mask_p, h_t, r_tm1)   # a_t : batch x n_dim
                                                                            # alpha : batch x T                                                                         
            alpha_vec[ix] = alpha
            m_t = torch.cat([a_t, h_t.cuda()], dim=-1)
            r_t, _ = self.m_gru(m_t.unsqueeze(0).to(device), 
                                r_tm1.unsqueeze(0).to(device))

            mask_t = mask_t.unsqueeze(1)  # batch x 1
            r_t = self.mask_mult(r_t[0], r_tm1, mask_t)
            r_tm1 = r_t

        return r_t, alpha_vec

    def forward(self, encoded_p, encoded_h, training):
        """Forward step
        
        Parameters
        ----
         - encoded_p (seq_len, batch, n_dim): encoding matrix premise
         - encoded_h (seq_len, batch, n_dim): encoding matrix of hypothesis
        """
        batch_size = encoded_p.size(1)

        mask_p = torch.any(torch.ne(encoded_p, 0), axis=2)
        mask_h = torch.any(torch.ne(encoded_h, 0), axis=2)
        
        encoded_p = self.compress(encoded_p)
        encoded_h = self.compress(encoded_h)

        encoded_p = F.dropout(encoded_p, p=self.options["DROPOUT"], training=training)
        encoded_h = F.dropout(encoded_h, p=self.options["DROPOUT"], training=training)

        # RNN
        h_p_0, h_n_0 = self._init_hidden(batch_size)  # 1 x batch x n_dim
        o_p, _ = self._gru_forward(self.premise_gru, encoded_p, mask_p, h_p_0)
        o_h, _ = self._gru_forward(self.hypothesis_gru, encoded_h, mask_h, h_n_0)

        # Attention
        r_0 = self._attn_gru_init_hidden(batch_size)
        h_star, _ = self._attn_gru_forward(o_h, mask_h, r_0, o_p, mask_p)

        # Output layer
        h_star = self.out(h_star.cuda())
        if self.options["LAST_NON_LINEAR"]:
            h_star = F.relu(h_star)

        return h_star