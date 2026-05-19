import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Conv1d(hidden_dim, dim, 1),
            nn.GELU(),
            nn.Dropout(dr)
        )
    def forward(self, x):
        return self.net(x)


class DSCNN(nn.Module):
    def __init__(self, embed_dim, num_patches, mlp_ratio, dr):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_mix_ffn = FeedForward(num_patches, int(num_patches*mlp_ratio), dr)
        self.patch_mix_ffn = FeedForward(embed_dim, int(embed_dim*mlp_ratio), dr)
        self.fusion = nn.Conv1d(num_patches, num_patches, 1)
    def forward(self, x):
        x = self.norm(x)
        x1 = self.embed_mix_ffn(x)
        x = x.transpose(1, 2)
        x2 = self.patch_mix_ffn(x)
        x2 = x2.transpose(1, 2)
        x_out = self.fusion(x1 + x2)
        return x_out