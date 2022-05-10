from importify import Serializable
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttnConfig(Serializable):
    def __init__(self, input_size):
        super(SelfAttnConfig, self).__init__()
        self.input_size = input_size
        self.hidden_dim = 512

class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttnConfig):
        super(SelfAttention, self).__init__()
        self.config = config

        self.w_query = nn.Linear(self.config.input_size, self.config.hidden_dim)
        self.w_key = nn.Linear(self.config.input_size, self.config.hidden_dim)
        self.w_value = nn.Linear(self.config.input_size, self.config.hidden_dim)

    def forward(self, x):
        # x: [b, l, input_size]
        # query: [b, l, hidden_dim]
        query = self.w_query(x)
        # key: [b, l, hidden_dim]
        key = self.w_key(x)
        # value: [b, l, hidden_dim]
        value = self.w_value(x)

        # q_k(query to key): [b, l, l]
        q_k = torch.matmul(query, torch.permute(key, (0, 2, 1)))
        q_k = F.softmax(q_k, dim=2)

        # q_v(query to value): [b, l, hidden_dim]
        q_v = torch.matmul(q_k, value)

        return q_v

class MultiHeadAttnConfig(Serializable):
    def __init__(self, attn_config: SelfAttnConfig):
        super(MultiHeadAttnConfig, self).__init__()
        self.n_head = 8
        self.attn_config = attn_config

class MultiHeadAttention(nn.Module):
    def __init__(self, config: MultiHeadAttnConfig):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.attns = nn.ModuleList([SelfAttention(config=self.config.attn_config)
                 for _ in range(self.config.n_head)])
        self.linear = nn.Linear(self.config.attn_config.input_size*self.config.n_head,
                                self.config.attn_config.input_size)

    def forward(self, x):
        # x: [b, l, input_size]
        heads = []
        for i in range(self.config.n_head):
            # res: [b, l, input_size]
            res = self.attns[i](x)
            heads.append(res)
        # x: [b, l, input_size*n_head]
        x = torch.cat(heads, 2) 
        # x: [b, l, input_size]
        x = self.linear(x)
        return x
        