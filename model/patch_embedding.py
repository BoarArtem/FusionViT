import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        num_patch = (img_size // patch_size)**2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # [1, 1, D]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1+num_patch, embed_dim))

    def forward(self, x):
        """
        :param x: [B, C, W, H]
        """
        B = x.size(0)

        # [B, D, W/P, H/P]
        x = self.proj(x)

        # flatten - [B, D, N]
        # transpose - [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        # [B, 1, D]
        cls_token = self.cls_token.expand(B, -1, -1)

        # [B, N, D] + [B, 1, D] = [B, N+1, D]
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed

        return x

