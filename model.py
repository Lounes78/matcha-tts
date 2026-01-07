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
            return
        
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  # (seq_len, d/2)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)  # (seq_len, d)

        # cache them
        self.cos_cahed = idx_theta2.cos()[:, None, None, :]  # (seq_len, 1, 1, d)
        self.sin_cached = idx_theta2.sin()[:, None, None, :]  # (seq_len, 1, 1, d)

    def _neg_half(self, x:torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
    
    def forward(self, x: torch.Tensor):
        """
        'x' is the Tensor at the head of a key or a query with shape [seq_len, batch_size, n_heads, d]
        """
        x = rearrange(x, 'b h t d -> t b h d')  # (seq_len, batch_size, n_heads, d)

        self._build_cache(x)

        x_rope, x_pass = x[..., :self.d], x[..., self.d:]  # (seq_len, batch_size, n_heads, d), (seq_len, batch_size, n_heads, d_pass)

        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cahed[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")
    
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            channels,
            out_channels,
            n_heads,
            heads_share=True,
            p_dropout=0.0,
            proximal_bias=False,
            proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv1d(channels, channels, kernel_size=1)

        self.query_rotary_pe = RotaryPositionalEmebeddings(self.k_channels*0.5)
        self.key_rotary_pe = RotaryPositionalEmebeddings(self.k_channels*0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, kernel_size=1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    
    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x 
    
    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if self.proximal_bias:
            assert t_s == t_t, "imal bias is only available for self-attention"
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn
    
    @staticmethod
    def _attention_bias_proximal(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0))
    


class FFN(nn.Module):
    def __ini__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
    
class Encoder(nn.Module):
    def __init__(
            self,
            hidden_channels,
            filter_channels, 
            n_heads,
            n_layers,
            kernel_size=1,
            p_dropout=0.0
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))
    
    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x 
    
class TextEncoder(nn.Module):
    def __init__(
        self, 
        encoder_type, 
        encoder_params,
        duration_predictor_params,
        n_vocab,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)

        if encoder_params.prenet:
                    self.prenet = ConvReluNorm(
                        self.n_channels,
                        self.n_channels,
                        self.n_channels,
                        kernel_size=5,
                        n_layers=3,
                        p_dropout=0.5,
                    )
        else:
            self.prenet = lambda x, x_mask: x

        self.encoder = Encoder(
            encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)
        self.proj_w = DurationPredictor(
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths, spks=None):
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            x (torch.Tensor): text input
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): average output of the encoder
                shape: (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): log duration predicted by the duration predictor
                shape: (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)
        mu = self.proj_m(x) * x_mask

        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask
    

# ### The Walkthrough

# 1. **Look up the meaning (`self.emb(x)`):**
# The input `x` comes in as a simple list of ID numbers (like "Word #45, Word #12"). The model looks these up in its internal dictionary to turn them into rich lists of numbers (vectors) that represent the *meaning* of each character or phoneme.
# 2. **Rotate the data (`transpose`):**
# The model prefers to work with the data sideways (features in the middle) compared to how it came in. This flips it to the correct orientation for the machinery that follows.
# 3. **Create a validity map (`x_mask`):**
# Since sentences have different lengths, we often add empty space (padding) to make them all fit in a batch. This step creates a "mask" that tells the model: "Pay attention to these real words, but completely ignore the empty space at the end."
# 4. **Initial Polish (`self.prenet`):**
# Before the heavy thinking starts, the data goes through a "Pre-Network." This is like a rough draft or a filter that smooths out the incoming data and gathers local details (like which letters are next to each other).
# 5. **Add Speaker Identity (`if self.n_spks > 1`):**
# If the model can speak in different voices, it needs to know *who* is talking. Here, it takes the "Speaker ID" (e.g., "Voice A") and attaches that tag to every single word in the sentence. Now the model knows: "Say 'Hello', and say it like Speaker A."
# 6. **Deep Thinking (`self.encoder`):**
# This is the "brain." It looks at the whole sentence at once. It figures out context—for example, knowing that the word "read" sounds different in "I will read" vs "I have read." It refines the understanding of the text.
# 7. **Predict Sound Features (`mu = self.proj_m`):**
# Now that it understands the text, it translates that understanding into acoustic features (the "ingredients" of sound, like tone and resonance) that the vocoder will later turn into audio.
# 8. **Freeze and Predict Timing (`detach` & `logw`):**
# * **`detach`:** It takes a snapshot of the encoder's understanding. It effectively says, "Don't let the next step change how we understood the text; just use this snapshot."
# * **`logw` (Duration):** Using that snapshot, it guesses how long each character should be spoken (e.g., the 'a' in 'cat' is short, the 'o' in 'cool' is long).


# ---
# ### Why `* math.sqrt(n_channels)`?

# This is a specific trick from the original "Attention Is All You Need" paper.
# **The Reason:**
# When the model looks up the word vectors (embeddings), those numbers are usually very small (close to 0 or 1). However, as the data flows deeper into the network, values tend to get added together and grow larger.
# If the starting numbers are too small, the signal gets "drowned out" or fades away before the model can use it effectively. Multiplying by the square root of the size (e.g., ) acts like an **amplifier**. It boosts the signal strength of the words right at the start so they remain significant throughout the entire process.