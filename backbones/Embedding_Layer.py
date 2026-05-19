import torch
import torch.nn as nn
from timm.layers import trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

class Embed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim, dr):
        super().__init__()
        self.patch_embed = PatchEmbed(seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dr)
        trunc_normal_(self.pos_embed, std=.02)
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x