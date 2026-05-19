import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

#Signal-wise Adaptive Filter
class SAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, N = x_fft.shape
        x_fft = x_fft.unsqueeze(-1)
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask

    def forward(self, x_in):
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_complex = x[:, 0, :] + 1j*x[:, 1, :]
        x_fft = torch.fft.fft(x_complex)

        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = (x_fft * freq_mask).to(x.device)
        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high

        x_weighted += x_weighted2

        x = torch.fft.ifft(x_weighted)

        x_real = torch.real(x).unsqueeze(1)
        x_imag = torch.imag(x).unsqueeze(1)
        x_comb = torch.cat((x_real, x_imag), dim=1).to(dtype)
        return x_comb

#Feature-wise Adaptive Filter
class FAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

        self.norm = nn.LayerNorm(dim)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, L = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x = self.norm(x)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = x_fft * freq_mask.to(x.device)

        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high

        x_weighted += x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, L)
        return x