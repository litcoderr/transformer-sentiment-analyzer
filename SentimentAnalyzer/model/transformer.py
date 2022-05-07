import torch
import torch.nn as nn

from . import ModelConfig

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()
        self.config = config

    def forward(self, x):
        return x