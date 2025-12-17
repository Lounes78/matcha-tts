import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from einops import rearrange, pack, repeat
import numpy as np
import math


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.var((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)  # (B, C, T)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape) # aka x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x
    

class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding= kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        
        # Layer 1: Created manually (Shape change: Input $\to$ Hidden)
        # Layers 2 to N: Created in the loop (Shape constant: Hidden $\to$ Hidden).
        # Activations: Defined once and reused everywhere because they have no weights.
        for i in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding= kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))

        # Mathematically, Conv1d(k=1) is identical to nn.Linear. However, the data shapes differ:
        # nn.Linear expects: (Batch, Time, Channels)
        # nn.Conv1d expects: (Batch, Channels, Time)
        self.proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x # (B, C, T)
        for i in range(self.n_layers):
            x = self.conv_layers[i](x*x_mask)  # (B, Hidden, T)
            x = self.norm_layers[i](x)         
            x = self.relu_drop(x)  # (B, Hidden, T)
        x = x_org + self.proj(x)
        return x * x_mask
    
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    # different from the precedent order in convrelunorm class
    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class RotaryPositionalEmebeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        """
        d is the number of features
        base is the constant used for calculating theta
        """
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cahed = None
        self.sin_cached = None 
    
    def _build_cache(self, x):
        """
        Cache cos and sin values
        """
        if self.cos_cahed is not None and x.shape[0] <= self.cos_cahed.shape[0]:
            
## Decoder : 
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb) ## ->> Créer une séquence 
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0) # -->  Multiplication entre les tenseurs
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # --> Concaténation des tenseurs (cos et sin)
        return emb
     
class Block1D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1), # Convolution 1D avec kernel size 3
            nn.GroupNorm(groups, dim_out),
            nn.Mish(), # Mish = x * tanh(softplus(x)
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask
    
class ResnetBlock1D(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out)) 
        self.block1 = Block1D(dim, dim_out, groups=groups) 
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output
