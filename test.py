import torch

def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0

        for inputs,labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = (correct/total)*100

        print(f"Total accuracy: {accuracy:.2f}%")