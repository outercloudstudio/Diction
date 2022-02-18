from turtle import color
import numpy
from sqlalchemy import false
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
epochs = 1000

training_dataloader, test_dataloader = Data.InitDataLoaders(batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = Model.Model()

model.load_state_dict(torch.load('diction_w2.pth'))
model.eval()

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train.Train(training_dataloader, model, loss_fn, optimizer, epochs)

# Test.Test(test_dataloader, model, loss_fn)

graphData = pd.read_csv("data3.csv")

betas = len(graphData.columns)
items = len(graphData.index)

colors = [
    "#08F7FE",
    "#FE53BB",
    "#F5D300",
    "#00FF41",
    "#eb344f"
]

plt.style.use('seaborn-dark')

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'

fig, (actual, predicted) = plt.subplots(2)

predictions = numpy.zeros((items, betas))

labels_map = [
    "New Features",
    "Fixes/Tweaks",
    "Parity",
    "Technical Update",
    "Gametest"
]

for i in range(betas - 1):
    print("Predicting " + str(i + 1) + " / " + str(betas - 1))
    gdata = graphData.iloc[:, i + 1].to_numpy(dtype=float)

    data = torch.from_numpy(gdata).float()

    print(data)

    pred = model(data.to(device))

    pred = pred.cpu().detach().numpy()

    for j in range(len(pred)):
        predictions[j][i + 1] = pred[j]

n_lines = 10
diff_linewidth = 1.05
alpha_value = 0.03

for i in range(items):
    data = graphData.iloc[i, 1:].to_numpy(dtype=float)

    data = numpy.append(data, 0)

    actual.plot(numpy.arange(betas), data, color=colors[i], marker='o')

    for n in range(1, n_lines+1):    
        actual.plot(numpy.arange(betas), data, color=colors[i], marker='o', linewidth=2+(diff_linewidth*n), alpha=alpha_value)

    print(data)

    actual.fill_between(x=numpy.arange(betas), y1=data, color=colors[i], alpha=0.1)

    predicted.plot(numpy.arange(betas), predictions[i], color=colors[i], marker='o', label=labels_map[i])

    for n in range(1, n_lines+1):    
        predicted.plot(numpy.arange(betas), predictions[i], color=colors[i], marker='o', linewidth=2+(diff_linewidth*n), alpha=alpha_value)

    predicted.fill_between(x=numpy.arange(betas), y1=predictions[i], color=colors[i], alpha=0.1)

fig.suptitle('Actual vs Preditcions', fontsize=14)

actual.grid(color='#2A3459', linewidth=2)
predicted.grid(color='#2A3459', linewidth=2)

predicted.legend(loc="upper left")

actual.plot()
predicted.plot()

plt.show()