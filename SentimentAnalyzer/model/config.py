from importify import Serializable

# Main Model Configuration
class ModelConfig(Serializable):
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.tokenizer_size = 42000

        # embedding
        self.embedding_dim = 32

        # Encoder
        self.encoder_config = EncoderConfig(input_size=self.embedding_dim)

        # Fully Connected
        self.fc_dim = 64

        # number of classes
        self.n_class = 2

# Encoder Configuration
class EncoderConfig(Serializable):
    def __init__(self, input_size):
        super(EncoderConfig, self).__init__()
        self.input_size = input_size
        self.n_layer = 8
        self.multi_head_attn = MultiHeadAttnConfig(
                                attn_config=SelfAttnConfig(
                                    input_size=self.input_size))

# Attention Configuration
class SelfAttnConfig(Serializable):
    def __init__(self, input_size):
        super(SelfAttnConfig, self).__init__()
        self.input_size = input_size
        self.hidden_dim = 32

class MultiHeadAttnConfig(Serializable):
    def __init__(self, attn_config: SelfAttnConfig):
        super(MultiHeadAttnConfig, self).__init__()
        self.n_head = 2
        self.attn_config = attn_config
