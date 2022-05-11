import os
from tqdm import tqdm
from importify import Serializable
import torch
import torch.nn as nn
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
    batch_label = torch.squeeze(batch_label, 1)
    batch_label = batch_label.long()
    return batch_tensor, batch_label

def validate_checkpoint(path):
    start_epoch_idx = 0
    if os.path.exists(path):
        # TODO check if valid
        start_epoch_idx = int(path.split("/")[-1].split(".ckpt")[0])
        return start_epoch_idx, path
    else:
        return start_epoch_idx, None

class TrainConfig(Serializable):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.config_file = None
        self.device = "cuda"
        # dataloader options
        self.batch_size = 32
        self.shuffle = True
        self.num_workers = 8
        
        # learning rate
        self.lr = 0.001

        # checkpoint option
        # where the training should start from
        self.checkpoint = 0
        self.save_rate = 10 # every (ckpt_rate) ammount of epoch, save ckpt and config file

        # test option
        self.test_rate = 10 # every (test_rate) ammount of epoch, perform testing

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
    test_dataset = NaverTestDataset()
    test_dataloader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn)

    # TODO implement checkpointing process
    start_epoch_idx, ckpt_path = validate_checkpoint(path=os.path.join(current_dir,
                                                  "SentimentAnalyzer",
                                                  "checkpoints",
                                                  "{}.ckpt".format(config.checkpoint)))
    
    app = App(model_config=config.model_config, checkpoint=ckpt_path, device=config.device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(app.model.parameters(), lr=config.lr)

    epoch_idx = start_epoch_idx
    while True:
        pbar = tqdm(train_dataloader)
        pbar.set_description("[train epoch #{}]".format(epoch_idx))
        for (tensor, label) in pbar:
            # TODO remove breaking after test impelmentation
            break

            # output: [b, n_classes]
            # label: [b]
            output = app.forward(tensor, is_train=True)

            # compute loss
            label = label.to(config.device)
            loss = criterion(input=output, target=label)

            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # TODO test using test dataset
        if epoch_idx % config.test_rate == 0:
            print("test")
            break
        # TODO save checkpoint / config
        # TODO implement monitoring usint Tensorboard

        epoch_idx += 1