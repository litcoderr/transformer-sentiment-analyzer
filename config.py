
from importify import Serializable
from SentimentAnalyzer.model.transformer import ModelConfig

class TrainConfig(Serializable):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.version = "v1.1_mini"
        
        self.config_file = ""
        self.device = "cuda"
        # dataloader options
        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 4
        
        # learning rate
        self.lr = 0.0001

        # checkpoint option
        # where the training should start from
        self.start_idx = 0
        self.ckpt_path = ""
        self.save_rate = 10 # every (ckpt_rate) ammount of epoch, save ckpt and config file

        # test option
        self.test_rate = 10 # every (test_rate) ammount of epoch, perform testing

        # model configuration
        self.model_config = ModelConfig()