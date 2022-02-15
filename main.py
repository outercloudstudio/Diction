import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

learning_rate = 1e-3
batch_size = 64
epochs = 10

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

model.to(device)

print("Loaded!")

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(test_data), size=(1,)).item()

    img, label = test_data[sample_idx]

    figure.add_subplot(rows, cols, i)

    X, Xlabel = test_data[sample_idx]

    X = X.to(device)

    logits = model(X)

    pred_probab = nn.Softmax(dim=1)(logits)

    y_pred = pred_probab.argmax(1)

    plt.title(labels_map[label] + " Guessed: " + labels_map[y_pred.item()])

    plt.axis("off")

    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)

#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)

#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)

#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# for t in range(epochs):
#    print(f"Epoch {t+1}\n-------------------------------")
#    train_loop(train_dataloader, model, loss_fn, optimizer)
#    test_loop(test_dataloader, model, loss_fn)
    
# print("Done!")

# torch.save(model.state_dict(), 'model_weights.pth')

# print("Saved!")