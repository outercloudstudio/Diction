from turtle import color
import numpy
import torch
from torch import nn, tensor
import matplotlib.pyplot as plt
import Data
import Model
import Train
import Test
import pandas as pd

#Stopped at 1.18.10.20

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

learning_rate = 1e-3
batch_size = 1
epochs = 10000

training_dataloader, test_dataloader = Data.InitDataLoaders(batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = Model.Model()

model.load_state_dict(torch.load('diction_w.pth'))
model.eval()

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Train.Train(training_dataloader, model, loss_fn, optimizer, epochs)

Test.Test(test_dataloader, model, loss_fn)

graphData = pd.read_csv("data2.csv")

betas = len(graphData.columns)
items = len(graphData.index)

colors = [
    "red",
    "blue",
    "black",
    "green",
    "yellow"
]

fig, (actual, predicted) = plt.subplots(2)

predictions = numpy.zeros((items, betas))

for i in range(betas - 1):
    print("Predicting " + str(i + 1) + " / " + str(betas - 1))
    gdata = graphData.iloc[:, i + 1].to_numpy(dtype=float)

    data = torch.from_numpy(gdata).float()

    print(data)

    pred = model(data.to(device))

    pred = pred.cpu().detach().numpy()

    for j in range(len(pred)):
        predictions[j][i + 1] = pred[j]

for i in range(items):
    data = graphData.iloc[i, 1:].to_numpy()

    data = numpy.append(data, 0)

    actual.plot(numpy.arange(betas), data, color=colors[i], marker='o')

    print(predictions[i])

    predicted.plot(numpy.arange(betas), predictions[i], color=colors[i], marker='o')

fig.suptitle('Actual vs Preditcions', fontsize=14)
actual.grid(True)

predicted.grid(True)

actual.plot()
predicted.plot()

plt.show()