import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class SnakeBeta(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta  = nn.Parameter(torch.ones(channels))
        self.eps = 1e-6

    def forward(self, x):
        # x shape: (B, T, C)
        alpha = self.alpha.view(1, 1, -1)
        beta  = self.beta.view(1, 1, -1)
        return x + (1.0 / (beta + self.eps)) * torch.sin(alpha * x) ** 2





class FeedForward(nn.Module):
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, final_dropout: bool = False):
        super().__init__()
        inner_dim = dim * mult

        self.fc1 = nn.Linear(dim, inner_dim)
        self.act = SnakeBeta(inner_dim)
        self.drop = nn.Dropout(dropout)

        self.fc2 = nn.Linear(inner_dim, dim)
        self.final_drop = nn.Dropout(dropout) if final_dropout else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.final_drop(x)
        return x




class BasicTransformerBlock(nn.Module):
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        
        # 1) Self-attention (Pre-LN)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop_attn(attn_out)

        # 2) Feed-forward (Pre-LN)
        x = x + self.ff(self.norm2(x))
        return x



