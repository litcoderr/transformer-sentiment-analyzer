import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, device):
        super(PositionalEncoding, self).__init__()
        self.max_len = 143
        self.embedding_dim = embedding_dim

        # encoding: [max_len, embedding_dim]
        self.encoding = torch.zeros(self.max_len, self.embedding_dim, device=device)
        self.encoding.requires_grad = False
        
        # pos: [max_len, 1] positional indexing vector
        pos = torch.arange(0, self.max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # i_rng: [embedding_dim//2]: ex) [0,2,4,...]
        i_rng = torch.arange(0, self.embedding_dim, step=2, device=device).float()

        # build encoding
        # trig_idx: [max_len, embedding_dim//2]
        trig_idx = pos / (10000 ** (i_rng / self.embedding_dim))
        self.encoding[:, ::2] = torch.sin(trig_idx)
        self.encoding[:, 1::2] = torch.cos(trig_idx)
        # encoding: [1, max_len, embedding_dim]
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # x: [b, seq_len, embedding_dim]
        batch_size, seq_len, embedding_dim = x.shape
        x += self.encoding[:,:seq_len,:]
        return x