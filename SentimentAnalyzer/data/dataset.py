import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .data_downloader import DATASET_DIR

PREPROCESSED_TRAIN_PATH = os.path.join(DATASET_DIR, "preprocessed", "train")
PREPROCESSED_TEST_PATH = os.path.join(DATASET_DIR, "preprocessed", "test")
DATA_PATH = {
    "train": {
        "raw": os.path.join(DATASET_DIR, "train.txt"),
        "processed": PREPROCESSED_TRAIN_PATH
    },
    "test": {
        "raw": os.path.join(DATASET_DIR, "test.txt"),
        "processed": PREPROCESSED_TEST_PATH
    },
}

def read_raw(path):
    with open(path, "r") as file:
        mem_str = file.read()
    processed = list(map(lambda r: tuple(r.split("\t")), mem_str.split("\n")[1:]))
    processed = [x for x in processed if len(x) == 3]
    return processed

class NaverDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.raw = read_raw(self.path["raw"])
    
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        # Tuple (id, text, label)
        (ids, text, label) = self.raw[idx]

        pickle_path = os.path.join(self.path["processed"], ids)
        with open(pickle_path, 'rb') as f:
            tensor = pickle.load(f)
        return (self.raw[idx], tensor)

class NaverTrainDataset(NaverDataset):
    def __init__(self):
        super().__init__(path=DATA_PATH["train"])

class NaverTestDataset(NaverDataset):
    def __init__(self):
        super().__init__(path=DATA_PATH["test"])

if __name__ == "__main__":
    def collate_fn(b):
        max_len = 0
        tensor_list = []
        label_list = []
        for ((ids, text, label), (tensor, _label)) in b:
            t: torch.Tensor = torch.Tensor(tensor)
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

    test_dataset = NaverTestDataset()
    dataloader = DataLoader(test_dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=collate_fn)

    for i_batch, batch in enumerate(dataloader):
        if i_batch == 1:
            break
        print(type(batch))
        print(batch[0].shape)
        print(batch[1].shape)