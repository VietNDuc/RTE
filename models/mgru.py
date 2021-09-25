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
        self.device = torch.device("cuda:0" if self.options["CUDA"] else "cpu")

        self.p_linear = nn.Linear(768, self.n_embed).to(self.device)
        self.h_linear = nn.Linear(768, self.n_embed).to(self.device)
        self.premise_gru = nn.GRU(self.n_embed, self.n_dim, bidirectional=False).to(self.device)
        self.hypothesis_gru = nn.GRU(self.n_embed, self.n_dim, bidirectional=False).to(self.device)
        self.out = nn.Linear(self.n_dim, self.n_out).to(self.device)
        
        # Attention Parameters
        self.W_s = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_dim, self.n_dim)).to(self.device))  # n_dim x n_dim
        self.register_parameter('W_s', self.W_s)
        self.W_t = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_dim, self.n_dim)).to(self.device))  # n_dim x n_dim
        self.register_parameter('W_t', self.W_t)
        self.w_e = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_dim, 1)).to(self.device)) # n_dim x 1
        self.register_parameter('w_e', self.w_e)
        self.W_m = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_dim, self.n_dim)).to(self.device))  # n_dim x n_dim
        self.register_parameter('W_m', self.W_m)

        # Match GRU parameters.
        self.m_gru = nn.GRU(self.n_dim + self.n_dim, self.n_dim, bidirectional=False).to(self.device)

    def _init_hidden(self, batch_size):
        """Init hidden matrix for GRU"""
        hidden_p = Variable(torch.ones(1, batch_size, self.n_dim))
        hidden_h = Variable(torch.ones(1, batch_size, self.n_dim))
        return hidden_p, hidden_h

    def _attn_gru_init_hidden(self, batch_size):
        """Init for GRU attention"""
        r_0 = Variable(torch.ones(batch_size, self.n_dim))
        return r_0
    
    def _gru_forward(self, gru, encoded_sent, h_0):
        """Stateless GRU for premise/hypothesis.

        Parameters:
        ----
         - gru: GRU cell
         - encoded_sent (seq_len, batch, n_embed) embedded matrix of premise/hypothesis sentence
         - h_0: init hidden vector for GRU cell

        Returns:
        ----
         - o_t (seq_len, batch, n_dim): output of GRU
         - h_t (1, batch, n_dim):  hidden state for t = seq_len
        """
        o_t, h_t = gru(encoded_sent.to(self.device), 
                       h_0.to(self.device))

        return o_t, h_t

    def _attention_forward(self, H_s, h_t, h_m_tm1=None):
        '''Word-by-word attention.

        Computes the Attention Weights over H_s using h_t (and h_m_tm1 if given)
        Returns an attention weighted representation of H_s, and the alphas.

        Parameters:
        ----
         - H_s (seq_len, batch, n_dim): output of all batchs come from GRU cell
         - h_t (batch, n_dim): hidden matrix for t-th word in hypothesis (batch)
         - h_m_tm1 (batch, n_dim): previous h_m

        Returns:
        ----
         - h_m (batch x n_dim)
         - alpha (batch x T): attention weight
        '''
        H_s = H_s.transpose(1, 0).to(self.device)

        Whs = torch.bmm(H_s, self.W_s.unsqueeze(0).expand(H_s.size(0), *self.W_s.size()))  # batch x seq x n_dim
        Wht = torch.mm(h_t.cuda(), self.W_t)  # batch x n_dim
        if h_m_tm1 is not None:
            W_r_tm1 = torch.mm(h_m_tm1.to(self.device), self.W_m)  # (batch, n_dim)
            Whs += W_r_tm1.unsqueeze(1)
        M = torch.tanh(Whs + Wht.unsqueeze(1).expand(Wht.size(0), H_s.size(1), Wht.size(1)))  # batch x seq x n_dim
        alpha = torch.bmm(M, self.w_e.unsqueeze(0).expand(H_s.size(0), *self.w_e.size())).squeeze(-1)  # batch x seq
        alpha = F.softmax(alpha)

        return torch.bmm(alpha.unsqueeze(1), H_s).squeeze(1), alpha

    def _attn_gru_forward(self, o_h, h_m_0, o_p):
        '''Use match-GRU to modeling the matching between the premise and the hypothesis.

        Parameters:
        ----
         - o_h (seq_len, batch, n_dim): hypothesis's output from GRU
         - h_m_0 (batch, n_dim): 
         - o_p (seq_len, batch,n_dim): premise's output from GRU; will attend on it at every step.

        Returns:
        ----
         - h_m_t (batch, n_dim) the last state of the rnn
         - alpha_vec (seq_len_p, batch, seq_len_h) the attn vec at every step
        '''
        seq_len_h = o_h.size(0)
        batch_size = o_h.size(1)
        seq_len_p = o_p.size(0)
        alpha_vec = Variable(torch.rand(seq_len_h, batch_size, seq_len_p))
        h_m_tm1 = h_m_0
        for ix, h_t in enumerate(o_h):
            #  h_t : batch x n_dim
            a_t, alpha = self._attention_forward(o_p, h_t, h_m_tm1)   # a_t: batch x n_dim; alpha: batch x seq_len                                                                         
            alpha_vec[ix] = alpha
            m_t = torch.cat([a_t, h_t.to(self.device)], dim=-1)
            h_m_t, _ = self.m_gru(m_t.unsqueeze(0).to(self.device), 
                                h_m_tm1.unsqueeze(0).to(self.device))
            h_m_tm1 = h_m_t[0]
            
        return h_m_t[0], alpha_vec

    def forward(self, encoded_p, encoded_h, training):
        """Forward step

        Parameters:
        ----
         - encoded_p (seq_len, batch, n_dim): encoding matrix premise
         - encoded_h (seq_len, batch, n_dim): encoding matrix of hypothesis
        """
        batch_size = encoded_p.size(1)
        
        encoded_p = F.dropout(encoded_p, p=self.options["DROPOUT"], training=training)
        encoded_h = F.dropout(encoded_h, p=self.options["DROPOUT"], training=training)
        # Firse linear layer: 768 -> n_embed
        encoded_p = self.p_linear(encoded_p)
        encoded_h = self.h_linear(encoded_h)
        # RNN
        h_p_0, h_h_0 = self._init_hidden(batch_size)  # 1 x batch x n_dim
        o_p, _ = self._gru_forward(self.premise_gru, encoded_p, h_p_0)
        o_h, _ = self._gru_forward(self.hypothesis_gru, encoded_h, h_h_0)
        # Attention
        h_m_0 = self._attn_gru_init_hidden(batch_size)
        h_star, _ = self._attn_gru_forward(o_h, h_m_0, o_p)
        # Output layer
        h_star = self.out(h_star.cuda())
        if self.options["LAST_NON_LINEAR"]:
            h_star = F.relu(h_star)

        return h_star