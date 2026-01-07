import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from einops import rearrange, pack, repeat
import numpy as np
from model.transformer import BasicTransformerBlock
from model.unet_blocks import ResnetBlock1D, Downsample1D, Upsample1D, SinusoidalPosEmb, TimestepEmbedding




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