from torch import nn

from model.encoder_layer import TransformerEncoderLayer
from model.patch_embedding import PatchEmbedding
from utils import get_device, get_criterion, get_optim


class FusionViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_classes, mlp_dim, depth, num_heads, dropout_val):
        super().__init__()

        self.path_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, mlp_dim, num_heads, dropout_val)
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.path_embed(x)
        x = self.encoder(x)
        cls_token = x[:, 0]

        return self.head(cls_token)

device = get_device()

model = FusionViT(32, 4, 3, 256, 100, 1024, 6, 8, 0.2).to(device)

criterion = get_criterion()
optim = get_optim(model)