from importify import Serializable
import torch
import torch.nn as nn

class ModelConfig(Serializable):
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.tokenizer_size = 42000

        # embedding
        self.embedding_dim = 512

        # Encoder
        self.encoder_config = EncoderConfig(input_size=self.embedding_dim)

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=self.config.tokenizer_size,
                                      embedding_dim=self.config.embedding_dim)
        self.encoder = Encoder(config=self.config.encoder_config)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.encoder(embedded)
        return output

class EncoderConfig(Serializable):
    def __init__(self, input_size):
        super(EncoderConfig, self).__init__()
        self.input_size = input_size

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()
        self.config = config

    def forward(self, x):
        # TODO implement encoder
        return x