from importify import Serializable
import torch
import torch.nn as nn

class SelfAttnConfig(Serializable):
    def __init__(self, input_size):
        super(SelfAttnConfig, self).__init__()
        self.input_size = input_size

class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttnConfig):
        super(SelfAttention, self).__init__()
        self.config = config

    def forward(self, x):
        # x: [b, l, input_size]
        return x

class MultiHeadAttnConfig(Serializable):
    def __init__(self, attn_config: SelfAttnConfig):
        super(MultiHeadAttnConfig, self).__init__()
        self.n_head = 8
        self.attn_config = attn_config

class MultiHeadAttention(nn.Module):
    def __init__(self, config: MultiHeadAttnConfig):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.attns = [SelfAttention(config=self.config.attn_config)
                 for _ in range(self.config.n_head)]
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
        