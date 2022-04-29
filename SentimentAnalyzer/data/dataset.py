import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .data_downloader import DATASET_DIR

class NaverDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, "r") as file:
            mem_str = file.read()
        processed = list(map(lambda r: tuple(r.split("\t")), mem_str.split("\n")[1:]))
        self.processed = [x for x in processed if len(x) == 3]
    
    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        # Tuple (id, text, label)
        return self.processed[idx]

class NaverTrainDataset(NaverDataset):
    def __init__(self):
        super(os.path.join(DATASET_DIR, "train.txt"))

class NaverTestDataset(NaverDataset):
    def __init__(self):
        super().__init__(os.path.join(DATASET_DIR, "test.txt"))

class NaverDataloader(DataLoader):
    pass


if __name__ == "__main__":
    test_dataset = NaverTestDataset()
    print(test_dataset.__getitem__(10))