import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# Import your custom modules
from model.utils import *
from model.text_encoder import TextEncoder  # Once you implement this
from model.transformer import BasicTransformerBlock, FeedForward, SnakeBeta
from model.decoder import Decoder
from model.unet_blocks import (
    LayerNorm, 
    ConvReluNorm, 
    DurationPredictor, 
    RotaryPositionalEmebeddings,
    SinusoidalPosEmb,
    Block1D,
    ResnetBlock1D,
    Downsample1D,
    Upsample1D,
    TimestepEmbedding
)
from model.flow_matching import FlowMatching  # Once you implement this