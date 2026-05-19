import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dr)
        )
    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, embed_dim, num_patches, mlp_ratio, dr):
        super().__init__()
        self.embedding_mix = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FeedForward(embed_dim, int(embed_dim*mlp_ratio), dr),
        )
    def forward(self, x):
        x = self.embedding_mix(x)
        return x