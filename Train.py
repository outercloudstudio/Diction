import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Train(dataloader, model, loss_fn, optimizer, epochs):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer)

    torch.save(model.state_dict(), 'diction_w.pth')

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)

            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")