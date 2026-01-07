"""
THIS FILE INCLUDES THE FULL CODE FOR THE MODEL IMPLEMENTATION OF MATCHA-TTS
INCLUDING:  
1 - Utility Functions Section
2 - Text Encoder Components Section
3 - Decoder / UNet Components Section
Each section is further divided into relevant subsections for better organization.

We used the following reference repository: https://github.com/shivammehta25/Matcha-TTS

Find on Docs/Matcha-tts One file Implementation the definition of the different components 
and who each one part works.

code by Arezki , Lounes , faycal , amira

"""



####################################################################################################
#                                         Library Imports Section
####################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from einops import rearrange, pack, repeat
import numpy as np
import math




####################################################################################################
#                                       1 - Utility Functions Section
####################################################################################################

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  1.1 Required utility functions  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

"from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/model.py" 

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length


def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


def normalize(data, mu, std):
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return (data - mu) / std


def denormalize(data, mu, std):
    if not isinstance(mu, float):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, float):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return data * std + mu


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  1.2 Alignment fallback  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>







####################################################################################################
#                                       2 - Text Encoder Components Section
####################################################################################################






#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  2.1 Basic Building Blocks  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


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



                     
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  2.2  Attention Components  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



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
    


#                             2.3 Main Text Encoder



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




####################################################################################################
#                                       3 - Decoder / UNet Components Section
####################################################################################################

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  3.1 UNet Building Blocks  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


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


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, dim, use_conv_transpose=True):
        super().__init__()
        self.use_conv_transpose = use_conv_transpose
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        else:
            self.conv = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, x):
        if self.use_conv_transpose:
            return self.conv(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
    
class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, act_fn="silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.Mish()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
    
    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample





                  
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 3.2 Transformer Components for UNet  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 




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


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 3.3  Main UNet  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>







class Decoder(nn.Module):
    

    def __init__(
        self,
        in_channels: int,
        mu_channels: int,
        out_channels: int,
        channels=(192, 256),           # UNet channel sizes per level
        num_res_blocks: int = 2,       # resnet blocks per level
        num_transformer_blocks: int = 1,
        num_heads: int = 4,            # transformer heads 
        time_emb_dim: int = 256,       # SinusoidalPosEmb dim
        time_mlp_dim: int = 512,       # TimestepEmbedding output dim
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mu_channels = mu_channels
        self.out_channels = out_channels
        self.channels = list(channels)

        #time embedding
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = TimestepEmbedding(time_emb_dim, time_mlp_dim, act_fn="silu")

        #input projection
        # concat([x, mu]) along channels -> project to channels[0]
        self.in_proj = nn.Conv1d(in_channels + mu_channels, self.channels[0], kernel_size=1)

        # down path
        self.down_resblocks = nn.ModuleList()
        self.down_transformers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_c = self.channels[0]
        for level, c in enumerate(self.channels):
            # at each level we work with channel size c
            if level == 0:
                c_in = prev_c
            else:
                c_in = prev_c

            # resnet blocks at this level
            res_list = nn.ModuleList()
            tr_list = nn.ModuleList()

            for b in range(num_res_blocks):
                res_list.append(ResnetBlock1D(c_in if b == 0 else c, c, time_mlp_dim))
                # transformer on (B,T,C) so channel size must match
                tr_list.append(
                    nn.ModuleList(
                        [
                            BasicTransformerBlock(
                                dim=c,
                                num_heads=num_heads,
                                dropout=dropout,
                                ffn_mult=ffn_mult,
                            )
                            for _ in range(num_transformer_blocks)
                        ]
                    )
                )

            self.down_resblocks.append(res_list)
            self.down_transformers.append(tr_list)

            prev_c = c

            # downsample after each level except last
            if level < len(self.channels) - 1:
                self.downsamples.append(Downsample1D(prev_c))

        # mid (bottleneck) ----
        mid_c = self.channels[-1]
        self.mid_res1 = ResnetBlock1D(mid_c, mid_c, time_mlp_dim)
        self.mid_tr = nn.ModuleList(
            [
                BasicTransformerBlock(mid_c, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
                for _ in range(num_transformer_blocks)
            ]
        )
        self.mid_res2 = ResnetBlock1D(mid_c, mid_c, time_mlp_dim)

        #up path
        self.upsamples = nn.ModuleList()
        self.up_resblocks = nn.ModuleList()
        self.up_transformers = nn.ModuleList()

        
        for level in reversed(range(len(self.channels))):
            c = self.channels[level]
            # upsample before processing level (except first in reverse, i.e., bottleneck level)
            if level < len(self.channels) - 1:
                self.upsamples.append(Upsample1D(prev_c))

            # after concatenation with skip: channels double -> resnet reduces back to c
            res_list = nn.ModuleList()
            tr_list = nn.ModuleList()

            for b in range(num_res_blocks):
                # first block sees concatenated skip
                in_c = (prev_c + c) if b == 0 else c
                res_list.append(ResnetBlock1D(in_c, c, time_mlp_dim))
                tr_list.append(
                    nn.ModuleList(
                        [
                            BasicTransformerBlock(c, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
                            for _ in range(num_transformer_blocks)
                        ]
                    )
                )

            self.up_resblocks.append(res_list)
            self.up_transformers.append(tr_list)

            prev_c = c

        # output projection
        self.out_norm = nn.GroupNorm(8, self.channels[0])
        self.out_act = nn.Mish()
        self.out_proj = nn.Conv1d(self.channels[0], out_channels, kernel_size=1)

    # helpers 
    @staticmethod
    def _mask_to_key_padding(mask_b1t: torch.Tensor) -> Optional[torch.Tensor]:
        
        if mask_b1t is None:
            return None
        m = mask_b1t.squeeze(1)  # (B,T)
        return (m == 0)

    @staticmethod
    def _downsample_mask(mask_b1t: torch.Tensor) -> torch.Tensor:
        
        return mask_b1t[:, :, ::2]

    @staticmethod
    def _upsample_mask(mask_b1t: torch.Tensor, target_len: int) -> torch.Tensor:
      
        m = mask_b1t.repeat_interleave(2, dim=2)
        if m.shape[-1] > target_len:
            m = m[:, :, :target_len]
        elif m.shape[-1] < target_len:
            pad = target_len - m.shape[-1]
            m = F.pad(m, (0, pad))
        return m

    def _apply_transformers(self, x_bct, mask_b1t, transformer_stack: nn.ModuleList):
    
        x = x_bct.transpose(1, 2)  # (B,T,C)
        kpm = self._mask_to_key_padding(mask_b1t)
        for blk in transformer_stack:
            x = blk(x, key_padding_mask=kpm)
        x = x.transpose(1, 2)      # (B,C,T)
        return x

   
    def forward(self, x, mask, mu, t):
        #build time embedding
        # t: (B,) -> (B, time_emb_dim) -> (B, time_mlp_dim)
        t_emb = self.time_pos_emb(t)
        t_emb = self.time_mlp(t_emb)

        #condition concat
        # x, mu: (B,*,T)  mask: (B,1,T)
        h = torch.cat([x, mu], dim=1)          # (B, in+mu, T)
        h = self.in_proj(h)                    # (B, ch0,   T)
        h = h * mask

        
        skips = []
        masks = [mask]

        for level in range(len(self.channels)):
            # res + transformer blocks
            for b in range(len(self.down_resblocks[level])):
                h = self.down_resblocks[level][b](h, mask, t_emb)  # (B,C,T)
                h = h * mask

                # transformer(s) for this block
                h = self._apply_transformers(h, mask, self.down_transformers[level][b])
                h = h * mask

                skips.append(h)

            # downsample (except last level)
            if level < len(self.channels) - 1:
                h = self.downsamples[level](h)                  # (B,C,T/2 approx)
                mask = self._downsample_mask(mask)              # (B,1,T/2)
                h = h * mask
                masks.append(mask)

      
        h = self.mid_res1(h, mask, t_emb); h = h * mask
        h = self._apply_transformers(h, mask, self.mid_tr); h = h * mask
        h = self.mid_res2(h, mask, t_emb); h = h * mask

      
        # modules in reverse level order.
        upsample_idx = 0

        for level_rev, level in enumerate(reversed(range(len(self.channels)))):
            # upsample before processing level (except first reverse step)
            if level < len(self.channels) - 1:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

                
        
                target_mask = masks[level]  # mask at this resolution
                mask = self._upsample_mask(mask, target_mask.shape[-1])
                
                if h.shape[-1] != target_mask.shape[-1]:
                    if h.shape[-1] > target_mask.shape[-1]:
                        h = h[:, :, : target_mask.shape[-1]]
                    else:
                        h = F.pad(h, (0, target_mask.shape[-1] - h.shape[-1]))
                mask = target_mask

            # pop skip and concat
            skip = skips.pop()
            # ensure same T
            if skip.shape[-1] != h.shape[-1]:
                # crop/pad skip to match
                if skip.shape[-1] > h.shape[-1]:
                    skip = skip[:, :, : h.shape[-1]]
                else:
                    skip = F.pad(skip, (0, h.shape[-1] - skip.shape[-1]))

            h = torch.cat([h, skip], dim=1)     # (B, prevC + C, T)

            # res + transformer blocks
            idx_level = level_rev  # because up_resblocks stored in same reverse order
            for b in range(len(self.up_resblocks[idx_level])):
                h = self.up_resblocks[idx_level][b](h, mask, t_emb)
                h = h * mask

                h = self._apply_transformers(h, mask, self.up_transformers[idx_level][b])
                h = h * mask

       
        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_proj(h) * mask
        return out





################################################################################################################
#                                       4. Flow Matching Section 
################################################################################################################


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  4.1 BASECFM(nn.Module) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



class BASECFM(nn.Module):
    """
    Classe de base pour le Conditional Flow Matching.
    Transforme du BRUIT en SPECTROGRAMME mel en suivant un "flux"
    progressif de t=0 (bruit pur) à t=1 (spectrogramme réel).

    """
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.get("solver", "euler")
        self.sigma_min = cfm_params.get("sigma_min", 1e-4)

    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        z = torch.randn_like(mu) * temperature
        dt = 1.0 / n_timesteps
        dt = torch.tensor([dt] * z.shape[0], dtype=z.dtype, device=z.device)
        # à vérfiier avec Arezki  mu, mask, spks
        if self.solver == "euler":
            for i in range(n_timesteps):
                t = i / n_timesteps
                t = torch.tensor([t] * z.shape[0], dtype=z.dtype, device=z.device)
                pred = self.estimator(z, mask, mu, t, spks, cond) # à voir avec Decoder de Amira
                z = z + pred * dt.unsqueeze(1).unsqueeze(1)
        
        elif self.solver == "midpoint":
            for i in range(n_timesteps):
                t = i / n_timesteps
                t = torch.tensor([t] * z.shape[0], dtype=z.dtype, device=z.device)
                pred = self.estimator(z, mask, mu, t, spks, cond) # à voir avec Decoder de Amira
                z_mid = z + pred * dt.unsqueeze(1).unsqueeze(1) * 0.5
                t_mid = t + dt * 0.5
                pred_mid = self.estimator(z_mid, mask, mu, t_mid, spks, cond) # à voir avec Decoder de Amira
                z = z + pred_mid * dt.unsqueeze(1).unsqueeze(1)
        
        else:
            raise NotImplementedError(f"Solver {self.solver} not implemented")
        
        return z


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  4.2 CFM(BASECFM) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class CFM(BASECFM):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator=None,
    ):
        super().__init__(
            n_feats,
            cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        
        in_channels = n_feats
        if estimator is None:
            raise ValueError("estimator must be provided")
        
        self.estimator = estimator # pareil à vérifier avec Amira

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        return super().forward(
            mu=mu,
            mask=mask,
            n_timesteps=n_timesteps,
            temperature=temperature,
            spks=spks,
            cond=cond,
        )

    def compute_loss(self, x1, mask, mu, spks=None, cond=None): 
        b, _, t = mu.shape

        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)

        y_t = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u_t = x1 - (1 - self.sigma_min) * z

        pred = self.estimator(y_t, mask, mu, t.squeeze(), spks, cond)

        loss = F.mse_loss(pred * mask, u_t * mask, reduction="sum") / (
            torch.sum(mask) * u_t.shape[1]
        )
        return loss, y_t, pred, u_t
    



####################################################################################################
#                                       5 - Main Model Class (Arezki)     
####################################################################################################

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 5. MatchaTTSModel >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>