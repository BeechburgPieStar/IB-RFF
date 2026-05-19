import torch
import torch.nn as nn
import torch.nn.functional as F

class ICNN(nn.Module):
    def __init__(self, embed_dim, num_patches, mlp_ratio, dr):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, int(embed_dim*mlp_ratio), 1)
        self.conv2 = nn.Conv1d(embed_dim, int(embed_dim*mlp_ratio), 3, 1, 1)
        self.conv3 = nn.Conv1d(int(embed_dim*mlp_ratio), embed_dim, 1)
        self.drop = nn.Dropout(dr)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x