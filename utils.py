import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_criterion():
    return CrossEntropyLoss()

def get_optim(model, lr=0.0003):
    return Adam(model.parameters(), lr)
