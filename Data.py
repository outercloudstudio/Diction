from numpy import dtype
import torch
from torchvision.transforms import Lambda
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BetaDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        self.labels = pd.read_csv(data_file).iloc[:, 2:]
        self.data = pd.read_csv(data_file).iloc[:, 1:-1]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #print("Giving size {}".format(len(self.labels) - 1))
        return len(self.labels) - 1

    def __getitem__(self, idx):
        #print("Getting item {}".format(idx))

        data = self.data.iloc[:, idx].to_numpy(dtype=float)
        label = self.labels.iloc[:, idx].to_numpy(dtype=float)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label

training_data = BetaDataset(data_file='data3.csv', transform=Lambda(lambda x: torch.from_numpy(x).float()), target_transform=Lambda(lambda y: torch.from_numpy(y).float()))
test_data = BetaDataset(data_file='data3.csv', transform=Lambda(lambda x: torch.from_numpy(x).float()), target_transform=Lambda(lambda y: torch.from_numpy(y).float()))

def InitDataLoaders(batch_size):
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print("Initialized Data Loader!")
    
    return train_dataloader, test_dataloader