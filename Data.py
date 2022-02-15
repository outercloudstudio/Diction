import torch
from torch import nn, tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class BetaDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        #Select all but first row
        self.labels = pd.read_csv(data_file).iloc[:, 2:]
        #Select all but first column and first row
        self.data = pd.read_csv(data_file).iloc[:, 1:-1]

        # print("Labels:")
        #print(self.labels)
        # #Gets second label
        # print(self.labels.iloc[1])
        # print("Data:")
        #print(self.data)
        # #Gets first row of data
        #print(self.data.iloc[:, 0])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data.iloc[:, idx].to_numpy()
        label = self.labels.iloc[:, idx].to_numpy()

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label

training_data = BetaDataset(data_file='data.csv', transform=Lambda(lambda x: torch.from_numpy(x)), target_transform=Lambda(lambda y: torch.from_numpy(y)))

print(training_data.__getitem__(0))