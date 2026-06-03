import torch

from dataset import train_loader, test_loader
from model.fusionvit import model, optim, criterion, device
from test import test


def train(model, num_epochs, optim, criterion, train_loader, device):
    print("FusionViT is starting training...")
    model.train()

    for epoch in range(num_epochs):
        training_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optim.step()

            training_loss += loss.item()

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {training_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), "fusionvit.pth")


if __name__ == "__main__":
    total_params = sum(p.numel() for p in model.parameters())
    num_epochs = 300

    print(f"Total params: {total_params:,}")
    train(model, num_epochs, optim, criterion, train_loader, device)
    test(model, device, test_loader)