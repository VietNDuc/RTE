import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomposableAttention(nn.Module):
    """An implementation of Decomposable Attention Model for Natural Language Inference 
    <https://arxiv.org/pdf/1606.01933v1.pdf>
    """
    def __init__ (self, options):
        super(DecomposableAttention, self).__init__()

        self.emded_dim = options['EMBED_DIM']  # 768 for base and 1024 for large bert
        self.dropout = options['DROPOUT']
        self.n_out = options['CLASSES']  # 3 classes for this task
        self.device = torch.device("cuda:0" if self.options["CUDA"] else "cpu")

        self.f = self.__mlp(self.emded_dim, self.emded_dim).to(self.device)
        self.g = self.__mlp(2*self.emded_dim, self.emded_dim).to(self.device)
        self.h = self.__mlp(2*self.emded_dim, self.emded_dim).to(self.device)
        
        self.out = nn.Linear(self.emded_dim, self.n_out).to(self.device)

    def __mlp(self, input_dim, output_dim):
        """Feed-forward neural network.
        """
        net = []
        net.append(nn.Dropout(p=self.dropout))
        net.append(nn.Linear(input_dim, output_dim, bias=True))
        net.append(nn.ReLU())
        net.append(nn.Dropout(p=self.dropout))
        net.append(nn.Linear(output_dim, output_dim, bias=True))
        net.append(nn.ReLU())       

        return nn.Sequential(*net)

    def _attend_block(self, p, h):
        """Attention phase.

        Calculate soft-align the elements of premise & hypothesis using a variant of neural attention 
        and then decompose the problem into the comparison of aligned subphrases.

        Params:
        ----
         - p, h (sent_len, batch_size, num_dim)

        Returns:
        ----
         - alpha (batch_size, len_h, n_dim)
         - beta (batch_size, len_p, n_dim)
        """
        f_p = self.f(p)
        f_h = self.f(h).transpose(1, 2)  # batch_size, sent_len, n_dim -> batch_size, n_dim, sent_len
        e_ij = torch.bmm(f_p, f_h)  # batch_size, len_p, len_h
        
        beta = torch.bmm(F.softmax(e_ij, dim=2), h)  # batch_size, len_p, n_dim
        alpha = torch.bmm(F.softmax(e_ij.transpose(1, 2), dim=2), p) # batch_size, len_h, n_dim
        
        return alpha, beta
        
    def _compare_block(self, p, h, alpha, beta):
        """Compare phase.

        Params:
        ----
         - p, h (sent_len, batch_size, num_dim)
         - alpha
         - beta

        Returns:
        ----
         - V_A, V_B

        """
        V_A = self.g(torch.cat([p, beta], dim=2))
        V_B = self.g(torch.cat([h, alpha], dim=2))
        
        return V_A, V_B

    def _aggregate_block(self, v_a, v_b):
        """Aggregate phase.

        Params:
        ----
         - v_a, v_b

        Returns:
        ----
         - y_hat

        """
        v1 = v_a.sum(dim=1)
        v2 = v_b.sum(dim=1)

        y_hat = self.h(torch.cat([v1, v2], dim=1))

        return y_hat
        
    def forward(self, encoded_p, encoded_h):
        """Forward step.

        Params:
        ----
         - encoded_p (seq_len, batch, n_dim): encoding matrix premise
         - encoded_h (seq_len, batch, n_dim): encoding matrix of hypothesis

        Returns:
        ----
         - out (batch, num_classes)
        """
        encoded_p = encoded_p.transpose(0, 1)
        encoded_h = encoded_h.transpose(0, 1)
        alpha, beta = self._attend_block(encoded_p, encoded_h)
        V_A, V_B = self._compare_block(encoded_p, encoded_h, alpha, beta)
        y_hat = self._aggregate_block(V_A, V_B)
        out = self.out(y_hat)
        
        return out