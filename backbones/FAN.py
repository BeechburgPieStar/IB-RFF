import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from .Adaptive_Filter_Module import SAF, FAF
from .Bottleneck_Layer.MLP import MLP
from .Bottleneck_Layer.ICNN import ICNN
from .Bottleneck_Layer.DSCNN import DSCNN
from .Embedding_Layer import Embed

class FAN(nn.Module):
    def __init__(
        self, seq_len=256, in_chans=2, patch_size=256, embed_dim=128,
        mlp_ratio=3.0, dr=0.5, num_classes=6, bottleneck_type = 'MLP', use_FAF=True, use_SAF=True, use_patch_mix=True, use_embedding_mix=True):
        super().__init__()
        self.use_FAF = use_FAF
        self.use_SAF = use_SAF

        if self.use_SAF:
            self.signal_denoising = SAF(seq_len)
        
        self.patch_embedding = Embed(seq_len, patch_size, in_chans, embed_dim, dr)
        
        if self.use_FAF:
            self.feature_denoising = FAF(embed_dim)

        # Bottleneck head
        num_patches = self.patch_embedding.patch_embed.num_patches
        if bottleneck_type == 'MLP':
            self.bottleneck = MLP(embed_dim, num_patches, mlp_ratio, dr)
        elif bottleneck_type == 'ICNN':
            self.bottleneck = ICNN(embed_dim, num_patches, mlp_ratio, dr)
        elif bottleneck_type == 'DSCNN':
            self.bottleneck = DSCNN(embed_dim, num_patches, mlp_ratio, dr)
        else:
            raise ValueError(f"Unsupported bottleneck_type: '{bottleneck_type}'. Must be one of ['MLP', 'ICNN', 'DSCNN'].")

        # Classification head
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_SAF:
            x = self.signal_denoising(x)

        x_patch = self.patch_embedding(x)

        x_res = x_patch

        if self.use_FAF:
            x_patch = self.feature_denoising(x_patch)

        x_neck = self.bottleneck(x_patch) + x_res

        x_mean = x_neck.mean(dim=1)

        cls_out = self.cls_head(x_mean)
        return x_neck, cls_out