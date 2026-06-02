from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_val):
        super().__init__()

        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=in_features
        )
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))

        return x