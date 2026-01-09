import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import lightning as L
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List
from pathlib import Path

# Ensure we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model import MatchaTTS
    from hifigan.meldataset import mel_spectrogram, load_wav
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure model.py and hifigan/ are in the same directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONSTANTS & SYMBOLS (From main.py)
# -----------------------------------------------------------------------------
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


# -----------------------------------------------------------------------------
# UTILS & ALIGNMENT (MAS)
# -----------------------------------------------------------------------------

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    factor = 2 ** num_downsamplings_in_unet
    # Round up to nearest multiple of factor
    if isinstance(length, torch.Tensor):
        return ((length + factor - 1) // factor) * factor
    return int(math.ceil(length / factor) * factor)

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not installed. MAS will be slow. Install with: pip install numba")

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _maximum_path_numba(paths, values, t_x_max, t_y_max):
        """Numba-optimized MAS matching Cython implementation."""
        b = values.shape[0]
        neg_inf = -1e9
        
        for k in prange(b):
            tx = t_x_max[k]
            ty = t_y_max[k]
            
            # Forward pass
            for y in range(ty):
                x_start = max(0, tx + y - ty)
                x_end = min(tx, y + 1)
                
                for x in range(x_start, x_end):
                    if x == y:
                        v_cur = neg_inf
                    else:
                        v_cur = values[k, x, y-1] if y > 0 else neg_inf
                    
                    if x == 0:
                        v_prev = 0.0 if y == 0 else neg_inf
                    else:
                        v_prev = values[k, x-1, y-1] if y > 0 else neg_inf
                    
                    values[k, x, y] = max(v_cur, v_prev) + values[k, x, y]
            
            # Backward pass
            index = tx - 1
            for y in range(ty - 1, -1, -1):
                paths[k, index, y] = 1
                if index != 0 and (index == y or values[k, index, y-1] < values[k, index-1, y-1]):
                    index = index - 1


def maximum_path(value, mask):
    """
    Monotonic Alignment Search.
    Uses Numba JIT if available (fast), otherwise falls back to pure Python (slow).
    
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    device = value.device
    dtype = value.dtype
    value = value * mask
    
    b, t_x, t_y = value.shape
    
    # Get actual lengths from mask
    t_x_max = mask.sum(dim=1)[:, 0].long()
    t_y_max = mask.sum(dim=2)[:, 0].long()
    
    if NUMBA_AVAILABLE:
        # Fast path: use numba
        values_np = value.cpu().numpy().astype(np.float32)
        paths_np = np.zeros_like(values_np, dtype=np.int32)
        t_x_np = t_x_max.cpu().numpy().astype(np.int32)
        t_y_np = t_y_max.cpu().numpy().astype(np.int32)
        
        _maximum_path_numba(paths_np, values_np, t_x_np, t_y_np)
        
        return torch.from_numpy(paths_np).to(device=device, dtype=dtype)
    else:
        # Slow fallback: pure Python
        neg_inf = -1e9
        path = torch.zeros(b, t_x, t_y, dtype=dtype, device=device)
        v = value.clone()
        
        for y in range(t_y):
            if y == 0:
                v[:, 1:, 0] = neg_inf
            else:
                v_cur = v[:, :, y-1].clone()
                if y < t_x:
                    v_cur[:, y] = neg_inf
                v_prev = torch.full((b, t_x), neg_inf, device=device, dtype=dtype)
                v_prev[:, 1:] = v[:, :-1, y-1]
                v[:, :, y] = torch.maximum(v_cur, v_prev) + value[:, :, y]
        
        for k in range(b):
            tx, ty = t_x_max[k].item(), t_y_max[k].item()
            if tx <= 0 or ty <= 0:
                continue
            index = tx - 1
            for y in range(ty - 1, -1, -1):
                path[k, index, y] = 1.0
                if index != 0 and y > 0:
                    if index == y or v[k, index, y-1] < v[k, index-1, y-1]:
                        index = index - 1
        return path


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# -----------------------------------------------------------------------------
# MODULE WRAPPER
# -----------------------------------------------------------------------------

class MatchaLightning(LightningModule):
    def __init__(self, model_params, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize internal model
        self.model = MatchaTTS(**model_params)
        self.learning_rate = learning_rate
        
        # Mel Params
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mels = 80
        self.fmin = 0
        self.fmax = 8000
        self.sampling_rate = 22050
        
        # Normalization Stats (LJSpeech)
        # These are crucial for Flow Matching to work well from N(0,1) prior
        self.register_buffer("mel_mean", torch.tensor(-5.5366))
        self.register_buffer("mel_std", torch.tensor(2.1161))
        
        # Update model buffers so they are saved in checkpoint
        self.model.mel_mean = self.mel_mean
        self.model.mel_std = self.mel_std

    def forward(self, x, x_lengths, y, y_lengths, spks=None):
        return self.model(x, x_lengths, y, y_lengths, spks)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        x, x_lengths = batch['text'], batch['text_lengths']
        spks = batch.get('spks', None)
        
        # Handle Speaker Embedding BEFORE encoder (important: encoder uses spk embedding!)
        if self.model.n_spks > 1 and spks is not None:
            spks = self.model.spk_emb(spks)
        else:
            spks = None
        
        # Generate Mel on GPU
        audio = batch['audio'] 
        
        # Audio is [B, T]. mel_spectrogram expects [B, T]
        y = mel_spectrogram(
            audio, 
            self.n_fft, 
            self.n_mels, 
            self.sampling_rate, 
            self.hop_length, 
            self.win_length, 
            self.fmin, 
            self.fmax, 
            center=False
        ) # [B, n_mels, T_mel]
        
        # Normalize Mel!
        y = (torch.log(torch.clamp(y, min=1e-5)) - self.mel_mean) / self.mel_std
        
        y_lengths = (batch['audio_lengths'] // self.hop_length).long() 
        
        
        # 1. Encoder Pass
        # mu: [b, c, t_x]
        # logw: [b, 1, t_x]
        # x_mask: [b, 1, t_x]
        mu, logw, x_mask = self.model.encoder(x, x_lengths, spks)
        
        # 2. Alignment / Duration
        
        with torch.no_grad():
            # Create masks
            y_max_length = y.shape[-1]
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)  # [b, 1, t_y] - same dtype as x_mask!
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)  # [b, 1, t_x, t_y]
            
            # Compute Log Prior (Log Likelihood of Gaussian with std=1)
            # Logic copied from original Matcha-TTS (matcha/models/matcha_tts.py)
            # log_prior = -0.5 * (y - mu)^2 + const
            
            const = -0.5 * math.log(2 * math.pi) * self.model.encoder.encoder_params.n_feats
            factor = -0.5 * torch.ones(mu.shape, dtype=mu.dtype, device=mu.device)
            
            # y_square = -0.5 * y^2
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            
            # y_mu_double = -0.5 * (-2 * mu * y) = mu * y
            y_mu_double = torch.matmul(2.0 * (factor * mu).transpose(1, 2), y)
            
            # mu_square = -0.5 * mu^2
            mu_square = torch.sum(factor * (mu**2), 1).unsqueeze(-1)
            
            # log_prior = -0.5*y^2 + mu*y - 0.5*mu^2 + const
            #           = -0.5 * (y - mu)^2 + const
            log_prior = y_square - y_mu_double + mu_square + const

            # Apply Maximum Path
            path = maximum_path(log_prior, attn_mask.squeeze(1)) # [b, t_x, t_y]

        # 3. Use Path to get Duration Targets
        # path sum over t_y gives duration for each t_x
        # We assume path is not empty for any text token if lengths match reasonably
        # Original: logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        # path is [b, t_x, t_y], sum over t_y -> [b, t_x], unsqueeze -> [b, 1, t_x]
        logw_ = torch.log(1e-8 + path.sum(dim=2).unsqueeze(1)) * x_mask
        
        # Duration Loss (original uses: torch.sum((logw - logw_) ** 2) / torch.sum(lengths))
        loss_duration = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        
        # 4. Upsample mu using path for CFM
        # path: [b, t_x, t_y]
        # mu: [b, c, t_x]
        # mu_y = mu * path
        # Checkpoint 1: Verified against original implementation using transposes explicitely
        # mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
        # Note: 'path' here is 'attn' in original code but squeeze not needed as it is [b,tx,ty] not [b,1,tx,ty] like orig attn
        # Our path is [b, t_x, t_y]
        # mu is [b, c, t_x]
        # User requested: mu_y = torch.matmul(path.transpose(1, 2), mu.transpose(1, 2)).transpose(1, 2)
        # path.T(1,2) -> [b, ty, tx]
        # mu.T(1,2)   -> [b, tx, c]
        # matmul      -> [b, ty, c]
        # .T(1,2)     -> [b, c, ty]
        # This is mathematically equivalent to mu @ path, but we apply the exact requested form for correctness verification.
        mu_y = torch.matmul(path.transpose(1, 2), mu.transpose(1, 2)).transpose(1, 2)
        
        # 5. CFM Loss
        # compute_loss calls estimator with mu_y (aligned)
        # Speaker embedding already computed at the start of training_step
            
        # compute_loss returns (loss, y_t, pred, u_t) - we only need loss
        cfm_loss, _, _, _ = self.model.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=None)
        
        # 6. Prior Loss (Added matching original implementation)
        # prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        # prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        # Re-calc manually:
        prior_loss = 0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
        prior_loss = prior_loss.sum() / (y_mask.sum() * self.model.encoder.encoder_params.n_feats)

        loss = cfm_loss + loss_duration + prior_loss
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/cfm_loss", cfm_loss)
        self.log("train/dur_loss", loss_duration)
        self.log("train/prior_loss", prior_loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Same as training_step but with val/ prefix for logging
        x, x_lengths = batch['text'], batch['text_lengths']
        spks = batch.get('spks', None)
        
        if self.model.n_spks > 1 and spks is not None:
            spks = self.model.spk_emb(spks)
        else:
            spks = None
        
        audio = batch['audio']
        y = mel_spectrogram(
            audio, self.n_fft, self.n_mels, self.sampling_rate,
            self.hop_length, self.win_length, self.fmin, self.fmax, center=False
        )
        y = (torch.log(torch.clamp(y, min=1e-5)) - self.mel_mean) / self.mel_std
        y_lengths = (batch['audio_lengths'] // self.hop_length).long()
        
        mu, logw, x_mask = self.model.encoder(x, x_lengths, spks)
        
        with torch.no_grad():
            y_max_length = y.shape[-1]
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            
            const = -0.5 * math.log(2 * math.pi) * self.model.encoder.encoder_params.n_feats
            factor = -0.5 * torch.ones(mu.shape, dtype=mu.dtype, device=mu.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const
            path = maximum_path(log_prior, attn_mask.squeeze(1))
        
        logw_ = torch.log(1e-8 + path.sum(dim=2).unsqueeze(1)) * x_mask
        loss_duration = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        
        mu_y = torch.matmul(path.transpose(1, 2), mu.transpose(1, 2)).transpose(1, 2)
        
        cfm_loss, _, _, _ = self.model.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=None)
        
        prior_loss = 0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
        prior_loss = prior_loss.sum() / (y_mask.sum() * self.model.encoder.encoder_params.n_feats)
        
        loss = cfm_loss + loss_duration + prior_loss
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/cfm_loss", cfm_loss, sync_dist=True)
        self.log("val/dur_loss", loss_duration, sync_dist=True)
        self.log("val/prior_loss", prior_loss, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        # Checkpoint 3: Optimizer
        # Original config configs/model/optimizer/adam.yaml specifies torch.optim.Adam (not AdamW)
        # weight_decay is 0.0 default
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class LJSpeechDataset(Dataset):
    def __init__(self, data_root, cache_in_memory=True):
        self.data_root = Path(data_root)
        self.wav_dir = self.data_root / "wavs"
        self.metadata_path = self.data_root / "metadata.csv"
        self.phonemes_path = self.data_root / "metadata_phonemes.csv"
        self.cache_in_memory = cache_in_memory
        self.audio_cache = {}

        # 1. Try Loading Pre-computed Phonemes
        self.precomputed_phonemes = {}
        if self.phonemes_path.exists():
            print(f"Loading pre-computed phonemes from {self.phonemes_path}...")
            with open(self.phonemes_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        self.precomputed_phonemes[parts[0]] = parts[1]
            self.global_phonemizer = None
            print(f"Loaded {len(self.precomputed_phonemes)} phonemized entries.")
        else:
            # Fallback to on-the-fly
            print("Pre-computed phonemes not found. Using on-the-fly phonemization (Slower).")
            try:
                import phonemizer
                self.global_phonemizer = phonemizer.backend.EspeakBackend(
                    language='en-us', preserve_punctuation=True, with_stress=True
                )
            except Exception as e:
                print(f"Warning: Could not load phonemizer: {e}")
                self.global_phonemizer = None
        
        if not self.metadata_path.exists():
             raise FileNotFoundError(f"Metadata not found at {self.metadata_path}.")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = [line.strip().split('|') for line in f]

        # 2. Pre-load Audio into RAM
        if self.cache_in_memory:
            print("Caching audio files into RAM (this speeds up training)...")
            from tqdm import tqdm
            # Filter metadata to existing files only to be safe
            valid_metadata = []
            for item in tqdm(self.metadata):
                file_id = item[0]
                wav_path = self.wav_dir / f"{file_id}.wav"
                if wav_path.exists():
                    audio_data, sr = load_wav(str(wav_path))
                    # Normalize float32
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data / 32768.0
                    self.audio_cache[file_id] = audio_data
                    valid_metadata.append(item)
            self.metadata = valid_metadata
            print(f"Cached {len(self.audio_cache)} audio files.")

            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        file_id = item[0]
        
        # A. Text Processing
        if file_id in self.precomputed_phonemes:
            # Use pre-computed
            phonemes = self.precomputed_phonemes[file_id]
            text_ints = [ _symbol_to_id.get(c, 0) for c in phonemes ]
        else:
            # Fallback
            text_str = item[2] if len(item) > 2 else item[1]
            if self.global_phonemizer:
                try:
                    phonemized_text = self.global_phonemizer.phonemize([text_str], strip=True)[0]
                    text_ints = [ _symbol_to_id.get(c, 0) for c in phonemized_text ]
                except Exception:
                    text_ints = [ _symbol_to_id.get(c, 0) for c in text_str ]
            else:
                 text_ints = [ _symbol_to_id.get(c, 0) for c in text_str ]

        # Apply intersperse
        text_ints = intersperse(text_ints, 0)
        text = torch.LongTensor(text_ints)
        
        # B. Audio Loading
        if self.cache_in_memory and file_id in self.audio_cache:
            audio_data = self.audio_cache[file_id]
        else:
            wav_path = self.wav_dir / f"{file_id}.wav"
            if not wav_path.exists():
                 return self.__getitem__((idx + 1) % len(self))
            audio_data, sr = load_wav(str(wav_path))
            if audio_data.dtype == np.int16:
                 audio_data = audio_data / 32768.0
                 
        audio = torch.from_numpy(audio_data).float() # [T_audio] (1D)


        spks = torch.tensor([0]).long()
        
        return {"text": text, "audio": audio, "spks": spks}

def collate_fn(batch):
    texts = [b['text'] for b in batch]
    audios = [b['audio'] for b in batch]
    spks = [b['spks'] for b in batch]
    
    text_lengths = torch.LongTensor([t.shape[0] for t in texts])
    audio_lengths = torch.LongTensor([a.shape[0] for a in audios])
    
    # Pad Text
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0) # [B, T_x]
    
    # Pad Audio
    # We should ensure audio length will produce mel length compatible with U-Net downsamplings
    # Though standard pad is fine, we handle mel length masking in model.
    audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0) # [B, T_audio]
    
    spks_padded = torch.stack(spks) if spks[0] is not None else None
    
    return {
        "text": texts_padded, 
        "text_lengths": text_lengths,
        "audio": audios_padded, 
        "audio_lengths": audio_lengths,
        "spks": spks_padded
    }

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def get_default_params():
    # From MAIN.PY
    # Checkpoint 5: Encoder type verified from configs/model/encoder/default.yaml ("RoPE Encoder")
    # This must match string check in TextEncoder class
    encoder_params = SimpleNamespace(
        encoder_type="RoPE Encoder", n_feats=80, n_channels=192, filter_channels=768,
        n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, prenet=True
    )
    # Decoder Params:
    # Based on Original Matcha-TTS configs/model/decoder/default.yaml
    decoder_params = SimpleNamespace(
        channels=(256, 256), 
        dropout=0.05, 
        attention_head_dim=64,
        n_blocks=1, 
        num_mid_blocks=2, 
        num_heads=2, 
        act_fn="snakebeta" 
    )
    cfm_params = {"solver": "euler", "sigma_min": 1e-4}
    duration_predictor_params = SimpleNamespace(
        filter_channels_dp=256, kernel_size=3, p_dropout=0.1
    )
    return {
        "n_vocab": len(symbols),
        "n_spks": 1,
        "spk_emb_dim": 64,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "cfm_params": cfm_params,
        "duration_predictor_params": duration_predictor_params
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count() if torch.cuda.is_available() else 0)
    parser.add_argument("--data_root", type=str, default="LJSpeech-1.1", help="Path to LJSpeech dataset root")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision (16-mixed, 32, etc)")
    parser.add_argument("--val_split", type=float, default=0.05, help="Fraction of data for validation (default 5%)")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional Lightning checkpoint to resume from. Can be a .ckpt file or a directory containing .ckpt files.",
    )
    args = parser.parse_args()

    def resolve_ckpt_path(ckpt_path: Optional[str]) -> Optional[str]:
        if not ckpt_path:
            return None

        path = Path(ckpt_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

        if path.is_file():
            if path.suffix != ".ckpt":
                raise ValueError(f"Checkpoint file must end with .ckpt: {path}")
            return str(path)

        ckpts = sorted(path.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in directory: {path}")
        return str(ckpts[0])

    # Params
    params = get_default_params()
    
    # Model
    model = MatchaLightning(params, learning_rate=args.lr)
    
    # Data
    if not os.path.exists(args.data_root):
        print(f"Data root {args.data_root} does not exist. Please download LJSpeech.")
        return

    # Effective batch size scaling
    batch_size = args.batch_size
    full_dataset = LJSpeechDataset(args.data_root)
    
    # Split into train/val
    from torch.utils.data import random_split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Auto-adjust workers: 4 GPUs -> likely powerful CPU. Use 16 workers.
    # Since we use RAM caching, workers don't need to do disk I/O, just tensor conversion.
    # We can reduce workers slightly to reduce CPU contest if caching is on.
    num_workers = 8 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Trainer
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger("lightning_logs", name="matcha_tts")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",  # Monitor validation loss
        mode="min", 
        every_n_epochs=2,
        save_top_k=3,  # Keep top 3 best checkpoints
        save_last=True,  # Also save last checkpoint
        filename="matcha-epoch{epoch:02d}-val_loss{val/loss:.2f}"
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=5.0, # Checkpoint 4: Added gradient clipping verified from config/trainer/default.yaml
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=5,
        val_check_interval=0.5,  # Validate every half epoch
    )
    
    print(f"Starting training on {args.gpus} GPUs with precision {args.precision}...")
    ckpt_path = resolve_ckpt_path(args.ckpt_path)
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
