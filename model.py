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

code by arezki , lounes , fayçal , amira

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
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

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
        self.cos_cached = None
        self.sin_cached = None 
    
    def _build_cache(self, x):
        """
        Cache cos and sin values
        """
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  # (seq_len, d/2)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)  # (seq_len, d)

        # cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]  # (seq_len, 1, 1, d)
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

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

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
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
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
        self.encoder_params = encoder_params 
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

class LoRACompatibleLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        # Equivalent to LoRACompatibleLinear(in_features, out_features)
        self.proj = nn.Linear(in_features, out_features)

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fn = None
        if activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)
        elif activation_fn == "gelu":
            act_fn = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        # Add others if needed, but matcha uses snakebeta/gelu usually

        self.net = nn.ModuleList([])
        if act_fn is not None:
            self.net.append(act_fn)
        
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        bias=False,
        cross_attention_dim=None,
        upcast_attention=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Hidden states: [Batch, Time, Dim] (Expected by Diffusers attention)
        # But Matcha passes (Batch, Time, Dim) because it permutes before calling.
        
        h = self.heads
        
        q = self.to_q(hidden_states)
        
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, "b t (h d) -> b h t d", h=h)
        k = rearrange(k, "b t (h d) -> b h t d", h=h)
        v = rearrange(v, "b t (h d) -> b h t d", h=h)

        # Scaled Dot-Product Attention
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if attention_mask is not None:
            # Mask handling: Matcha passes (Batch, Time) mask.
            # We need to broadcast it to (Batch, Heads, Time, Time) or similar
            # If attention_mask is (B, T), we mask the key positions (j dimension)
            if attention_mask.ndim == 2:
                # attention_mask: 1 = keep, 0 = mask
                # Convert to large negative value for masking
                mask = attention_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, T)
                sim = sim.masked_fill(mask == 0, -torch.finfo(sim.dtype).min)

        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
    ):
        super().__init__()
        
        # Norm 1 -> Self Attn
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # Norm 3 -> Feed Forward (Skipping 2 as per Matcha default)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(self, hidden_states, attention_mask=None, timestep=None):
        # 1. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = attn_output + hidden_states

        # 3. Feed Forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block1D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
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
    def __init__(self, channels, use_conv_transpose=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv_transpose = use_conv_transpose
        
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        else:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        if self.use_conv_transpose:
            return self.conv(inputs)
        # Fallback for interpolate
        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return self.conv(outputs)

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, act_fn="silu", out_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.Mish()
        
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        time_emb_dim=None,  # Not used in original signature explicitly but used in logic above
        time_mlp_dim=None,  # Not used in original signature explicitly
        # Compat args to match user's explicit call:
        ffn_mult=4,
        **kwargs # Catch-all for extra params
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time Embedding Setup (Matching Original)
        # Note: in_channels passed here is 2*n_feats (160) typically
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        
        # Logic from original Decoder: time_embed_dim = channels[0] * 4
        # But wait, original code uses 'time_embed_dim' var inside init
        inner_time_embed_dim = channels[0] * 4 
        
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=inner_time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        
        # Down Blocks
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=inner_time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn="snakebeta", 
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # Mid Blocks
        for i in range(num_mid_blocks):
            # Input to mid block is the output of the last down block (channels[-1])
            input_channel = channels[-1] 
            
            resnet = ResnetBlock1D(dim=input_channel, dim_out=input_channel, time_emb_dim=inner_time_embed_dim)
            
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=input_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn="snakebeta",
                    )
                    for _ in range(n_blocks)
                ]
            )
            
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # Up Blocks
        # channels = channels[::-1] + (channels[0],) -> [256, 256, 256] if [256, 256]
        # logic from original:
        reversed_channels = list(channels[::-1]) + [channels[0]]
        
        for i in range(len(reversed_channels) - 1):
            input_channel = reversed_channels[i]
            output_channel = reversed_channels[i + 1]
            is_last = i == len(reversed_channels) - 2

            # Input dimension is 2 * input_channel because of concatenation with skip connection
            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=inner_time_embed_dim,
            )
            
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn="snakebeta",
                    )
                    for _ in range(n_blocks)
                ]
            )
            
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        # x: (B, C, T)  (No input projection layer here, handled by first block)
        # But we need to combine x and mu. 
        # Matcha logic: x = pack([x, mu], "b * t")[0] -> Interleave channels? 
        # Actually pack 'b * t' merges first dimension? No.
        # einops pack([a, b], 'b * t') means concat along the * dimension (channels).
        
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # Concat x and mu along channel dimension
        x = torch.cat([x, mu], dim=1) 
        
        if spks is not None:
             spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
             x = torch.cat([x, spks], dim=1)

        hiddens = []
        masks = [mask]
        
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            
            # Transformer needs (Batch, Time, Channels)
            x = rearrange(x, "b c t -> b t c")
            mask_down_t = rearrange(mask_down, "b 1 t -> b t")
            
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down_t,
                    timestep=t,
                )
            
            x = rearrange(x, "b t c -> b c t")
            
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid_t = rearrange(mask_mid, "b 1 t -> b t")
            
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid_t,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")

        for i, (resnet, transformer_blocks, upsample) in enumerate(self.up_blocks):
            mask_up = masks.pop()
            skip = hiddens.pop()
            
            # Pack/Concat x and skip
            # Handle potential 1-px mismatches from previous block's upsampling (odd vs even lengths)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")

            x = torch.cat([x, skip], dim=1)
            
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up_t = rearrange(mask_up, "b 1 t -> b t")
            
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up_t,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask





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

class MatchaTTS(nn.Module):
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        encoder_params,
        decoder_params,
        cfm_params,
        duration_predictor_params,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        
        if n_spks > 1:
            self.spk_emb = nn.Embedding(n_spks, spk_emb_dim)

        # Register buffers for mel_mean and mel_std to handle checkpoint loading
        self.register_buffer("mel_mean", torch.tensor(0.0))
        self.register_buffer("mel_std", torch.tensor(1.0))

        # 1. Initialize Text Encoder
        self.encoder = TextEncoder(
            encoder_type=encoder_params.encoder_type,
            encoder_params=encoder_params,
            duration_predictor_params=duration_predictor_params,
            n_vocab=n_vocab,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # 2. Initialize Decoder (The estimator)
        # Note: in_channels must be 2 * n_feats because we concatenate noisy mel (80) + conditional mel (80)
        
        decoder_in_channels = 2 * encoder_params.n_feats
        if n_spks > 1:
            decoder_in_channels += spk_emb_dim

        decoder_module = Decoder(
            in_channels=decoder_in_channels,
            out_channels=encoder_params.n_feats,
            channels=decoder_params.channels,
            dropout=decoder_params.dropout,
            attention_head_dim=decoder_params.attention_head_dim,
            n_blocks=decoder_params.n_blocks,
            num_mid_blocks=decoder_params.num_mid_blocks,
            num_heads=decoder_params.num_heads,
            act_fn=decoder_params.act_fn,
        )

        # 3. Initialize CFM (Flow Matching) - This is 'self.decoder' in the checkpoint
        self.decoder = CFM(
            n_feats=encoder_params.n_feats,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            estimator=decoder_module, 
        )

    def forward(self, x, x_lengths, y, y_lengths, spks=None):
        """
        Training forward pass.
        x: Text input sequences
        y: Mel-spectrogram target
        """
        # 1. Get Encoder Outputs (mu for alignment, logw for duration)
        mu, logw, x_mask = self.encoder(x, x_lengths, spks)

        # 2. Compute Duration Loss
        # (Assuming you align y to x or align lengths externally. 
        # In full Matcha, alignment is usually computed here or provided)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w)
        y_lengths_pred = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        
        # 3. Compute Flow Matching Loss
        # We align 'mu' to the length of 'y' for training using the mask
        # (Simplification: In training, we often use ground truth lengths or alignment search)
        
        # Resize mu to match y for CFM calculation (masking/padding handled inside)
        if mu.shape[-1] != y.shape[-1]:
             # In a real training loop, you would use Monotonic Alignment Search (MAS) here
             # For this implementation, we assume lengths are compatible or aligned
             pass 

        cfm_loss, _, _, _ = self.decoder.compute_loss(y, x_mask, mu, spks, cond=None)
        
        return cfm_loss, logw, y_lengths_pred

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        """
        Inference / Generation
        """
        # 1. Encode Text
        mu, logw, x_mask = self.encoder(x, x_lengths, spks)

        # 2. Predict Durations
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

        # 3. Upsample mu to match audio duration
        y_max_length = y_lengths.max()
        
        # Needed for UNet compatibility (must be divisible by downsample factor)
        y_max_length_ = fix_len_compatibility(y_max_length)
        
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # 4. Run Flow Matching to generate Mel-Spectrogram
        mel = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks, cond=None)
        
        # Denormalize (Important for Vocoder!)
        mel = denormalize(mel, self.mel_mean, self.mel_std)
        
        # Crop back to original length if needed (though usually we decode the padded version)
        mel = mel[:, :, :y_max_length]
        
        return mel, y_lengths
    

