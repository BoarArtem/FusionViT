import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_criterion():
    return CrossEntropyLoss()

def get_optim(model, lr=0.0001):
    return AdamW(model.parameters(), lr, weight_decay=0.05)
