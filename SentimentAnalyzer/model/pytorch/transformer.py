import torch
import torch.nn as nn
from collections import OrderedDict

from ..config import ModelConfig, EncoderConfig
from .encoding import PositionalEncoding
from .attn import MultiHeadAttention, MultiHeadAttnConfig, SelfAttnConfig, ResidualMultiHeadAttention

class TransformerV1(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super(TransformerV1, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=self.config.tokenizer_size,
                                      embedding_dim=self.config.embedding_dim)
        self.positional = PositionalEncoding(embedding_dim=self.config.embedding_dim, device=device)
        self.encoder = Encoder(config=self.config.encoder_config)
        self.fc1 = nn.Linear(self.config.embedding_dim, self.config.fc_dim)
        self.fc2 = nn.Linear(self.config.fc_dim, self.config.n_class)

    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoded = self.positional(embedded)
        encoder_output = self.encoder(positional_encoded)
        encoder_output = torch.mean(encoder_output, dim=1)
        latent = self.fc1(encoder_output)
        logit = self.fc2(latent)
        return logit

class TransformerV1_1(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super(TransformerV1_1, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=self.config.tokenizer_size,
                                      embedding_dim=self.config.embedding_dim)
        self.positional = PositionalEncoding(embedding_dim=self.config.embedding_dim, device=device)
        self.encoder = Encoder(config=self.config.encoder_config)
        self.fc1 = nn.Linear(self.config.embedding_dim, self.config.fc_dim)
        self.fc2 = nn.Linear(self.config.fc_dim, self.config.n_class)

    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoded = self.positional(embedded)
        encoder_output = self.encoder(positional_encoded)
        encoder_output = encoder_output[:,-1,:]
        latent = self.fc1(encoder_output)
        logit = self.fc2(latent)
        return logit

class TransformerV1_residual(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super(TransformerV1_residual, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=self.config.tokenizer_size,
                                      embedding_dim=self.config.embedding_dim)
        self.positional = PositionalEncoding(embedding_dim=self.config.embedding_dim, device=device)
        self.encoder = ResidualEncoder(config=self.config.encoder_config)
        self.fc1 = nn.Linear(self.config.embedding_dim, self.config.fc_dim)
        self.fc2 = nn.Linear(self.config.fc_dim, self.config.n_class)

    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoded = self.positional(embedded)
        encoder_output = self.encoder(positional_encoded)
        encoder_output = encoder_output[:,-1,:]
        latent = self.fc1(encoder_output)
        logit = self.fc2(latent)
        return logit

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()
        self.config = config

        self.attn_layers = nn.Sequential(OrderedDict([
            ("AttnLayer_{}".format(i), MultiHeadAttention(config=self.config.multi_head_attn))
            for i in range(self.config.n_layer)
        ]))

    def forward(self, x):
        x = self.attn_layers(x)
        return x

class ResidualEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super(ResidualEncoder, self).__init__()
        self.config = config

        self.attn_layers = nn.Sequential(OrderedDict([
            ("AttnLayer_{}".format(i), ResidualMultiHeadAttention(config=self.config.multi_head_attn))
            for i in range(self.config.n_layer)
        ]))

    def forward(self, x):
        x = self.attn_layers(x)
        return x