import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
import matplotlib.pyplot as plt
import Data
import Model
import Train
import Test

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

learning_rate = 1e-3
batch_size = 64
epochs = 10

training_dataloader, test_dataloader = Data.InitDataLoaders(batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = Model.Model().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

Train.Train(training_dataloader, model, loss_fn, optimizer, epochs)

Test.Test(test_dataloader, model, loss_fn)