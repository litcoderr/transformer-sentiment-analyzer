import os
from pathlib import Path
import secrets
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from SentimentAnalyzer import App
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

    # validate checkpoint
    start_epoch_idx = config.start_idx
    
    app = App(config=config, device=config.device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(app.model.parameters(), lr=config.lr)

    # tensorboard
    log_dir = os.path.join(current_dir, "tensorlog", config.version)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    epoch_idx = start_epoch_idx
    while True:
        n_train_iter = len(train_dataloader)
        pbar = tqdm(train_dataloader)
        pbar.set_description("[train epoch #{}]".format(epoch_idx))

        total_loss = 0
        total_acc = 0
        for i, (tensor, label) in enumerate(pbar):
            label = label.to(config.device)
            # output: [b, n_classes]
            # label: [b]
            output = app.forward(tensor, is_train=True)

            # compute accuracy
            # argmax: [b]
            argmax = torch.squeeze(torch.argmax(output.detach(), dim=1, keepdim=True), 1)
            # compared: [b]
            compared = torch.eq(argmax, label.detach())
            N = compared.shape[0]
            true_pos = torch.sum(compared)
            total_acc += true_pos.detach()

            # compute loss
            loss = criterion(input=output, target=label)
            total_loss += loss.detach()

            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_acc = total_acc.detach().cpu().numpy()
        total_loss = total_loss.detach().cpu().numpy()
        total_loss /= len(train_dataset) 
        total_acc = (total_acc / len(train_dataset)) * 100

        writer.add_scalar("Loss/train", total_loss, epoch_idx)
        writer.add_scalar("Acc/train", total_acc, epoch_idx)
            
        # save checkpoint and config file
        if epoch_idx % config.save_rate == 0:
            base_ckpt_dir = os.path.join(current_dir, "SentimentAnalyzer", "checkpoints")
            config_dir = os.path.join(base_ckpt_dir, "{}_{}.json".format(config.version, str(epoch_idx).zfill(5)))
            ckpt_dir = os.path.join(base_ckpt_dir, "{}.ckpt".format(str(secrets.token_hex(nbytes=8))))

            # save checkpoint
            torch.save({
                "checkpoint": app.model.state_dict()
            }, ckpt_dir)
            # save config file
            config.start_idx = epoch_idx+1
            config.ckpt_path = ckpt_dir
            config.export_json(path=config_dir, ignore_error=True)

        # test using test dataset
        if epoch_idx % config.test_rate == 0:
            with torch.no_grad():
                pbar = tqdm(test_dataloader)
                pbar.set_description("[test]")

                total_loss = 0
                total_acc = 0
                for (tensor, label) in pbar:
                    # output: [b, n_classes]
                    # label: [b]
                    output = app.forward(tensor, is_train=False)
                    label = label.to(config.device)

                    # compute loss
                    loss = criterion(input=output, target=label)
                    total_loss += loss.detach()

                    # compute accuracy
                    # argmax: [b]
                    argmax = torch.squeeze(torch.argmax(output.detach(), dim=1, keepdim=True), 1)
                    # compared: [b]
                    compared = torch.eq(argmax, label.detach())
                    N = compared.shape[0]
                    true_pos = torch.sum(compared.detach())
                    total_acc += true_pos.detach()
                total_acc = total_acc.detach().cpu().numpy()
                total_loss /= len(test_dataset) 
                total_acc = (total_acc / len(test_dataset)) * 100

                writer.add_scalar("Loss/test", total_loss, epoch_idx)
                writer.add_scalar("Acc/test", total_acc, epoch_idx)

        epoch_idx += 1
    writer.close()