import os
from tqdm import tqdm
from importify import Serializable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SentimentAnalyzer import App
from SentimentAnalyzer.model.transformer import ModelConfig
from SentimentAnalyzer.data.dataset import NaverTrainDataset, NaverTestDataset

def collate_fn(b):
    max_len = 0
    tensor_list = []
    label_list = []
    for ((ids, text, label), (tensor, _label)) in b:
        t: torch.Tensor = torch.tensor(tensor, dtype=torch.int32)
        if max_len < t.shape[0]:
            max_len = t.shape[0]
        tensor_list.append(t)
        label_tensor = torch.Tensor([_label])
        label_list.append(torch.unsqueeze(label_tensor, 0))

    padded_list = []
    for t in tensor_list:
        t = torch.unsqueeze(t, 0)
        padded = F.pad(t, (0,max_len-t.shape[1]), "constant", 0)
        padded_list.append(padded)
    batch_tensor = torch.cat(padded_list, 0) 

    batch_label = torch.cat(label_list, 0)
    return batch_tensor, batch_label

def validate_checkpoint(path):
    start_batch_idx = 0
    if os.path.exists(path):
        # TODO check if valid
        start_batch_idx = int(path.split("/")[-1].split(".ckpt")[0])
        return start_batch_idx, path
    else:
        return start_batch_idx, None

class TrainConfig(Serializable):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.config_file = None
        self.device = "cuda"
        # dataloader options
        self.batch_size = 16
        self.shuffle = True
        self.num_workers = 8

        # checkpoint option
        # where the training should start from
        self.checkpoint = 0

        # model configuration
        self.model_config = ModelConfig()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))

    config = TrainConfig()
    config.parse()
    if config.config_file:
        _, config = TrainConfig.import_json(path=config.config_file)

    train_dataset = NaverTrainDataset()
    train_dataloader = DataLoader(train_dataset,
                            batch_size=config.batch_size,
                            shuffle=config.shuffle,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn)

    # TODO implement checkpointing process
    start_batch_idx, ckpt_path = validate_checkpoint(path=os.path.join(current_dir,
                                                  "SentimentAnalyzer",
                                                  "checkpoints",
                                                  "{}.ckpt".format(config.checkpoint)))
    
    app = App(model_config=config.model_config, checkpoint=ckpt_path, device=config.device)

    batch_idx = start_batch_idx
    while True:
        pbar = tqdm(train_dataloader)
        for (tensor, label) in pbar:
            pbar.set_description("[batch #{}]".format(batch_idx))

            output = app.forward(tensor, is_train=True)

            # TODO compute gradient
            # TODO back propagate
            break
        break
            
        batch_idx += 1

        # TODO save checkpoint / config