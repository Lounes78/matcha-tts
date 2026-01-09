"""
Matcha-TTS Training Script - Fully Standalone Version
"""
import argparse
import os
import sys
import math
import random
import re
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as ta
from lightning import LightningDataModule
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data.dataloader import DataLoader
from librosa.filters import mel as librosa_mel_fn
from unidecode import unidecode
from types import SimpleNamespace

try:
    import phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("Warning: phonemizer not available")

# Import local model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MatchaTTS

# Import Numba for fast monotonic alignment
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available, using slow Python fallback")


# ============================================================================
# Text Processing - Symbols and Cleaners
# ============================================================================

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


# Text cleaners
_whitespace_re = re.compile(r"\s+")
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
        ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
        ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ]
]

if PHONEMIZER_AVAILABLE:
    critical_logger = logging.getLogger("phonemizer")
    critical_logger.setLevel(logging.CRITICAL)
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="en-us",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=critical_logger,
    )


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion + phonemization"""
    if not PHONEMIZER_AVAILABLE:
        # Fallback: just return cleaned text without phonemization
        text = convert_to_ascii(text)
        text = lowercase(text)
        text = expand_abbreviations(text)
        text = collapse_whitespace(text)
        return text
    
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def text_to_sequence(text, cleaner_names):
    """Convert text to sequence of symbol IDs"""
    sequence = []
    clean_text = text
    
    for cleaner_name in cleaner_names:
        if cleaner_name == "english_cleaners2":
            clean_text = english_cleaners2(clean_text)
        else:
            clean_text = lowercase(clean_text)
            clean_text = collapse_whitespace(clean_text)
    
    for symbol in clean_text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        # Skip unknown symbols
    
    return sequence, clean_text


def intersperse(lst, item):
    """Adds blank symbol between characters"""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


# ============================================================================
# Audio Processing
# ============================================================================

mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Convert audio to mel spectrogram"""
    if torch.min(y) < -1.0:
        print("Warning: min audio value is", torch.min(y))
    if torch.max(y) > 1.0:
        print("Warning: max audio value is", torch.max(y))

    global mel_basis, hann_window
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    
    return spec


def normalize(data, mu, std):
    """Normalize data with mean and std"""
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


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    """Ensure length is compatible with UNet downsampling"""
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length


# ============================================================================
# Monotonic Alignment Search (MAS) - Numba-optimized
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def maximum_path_jit(paths, values, t_ys, t_xs):
        """Numba JIT-compiled version for speed (600x faster)"""
        b = paths.shape[0]
        for i in range(b):
            path = paths[i, :t_xs[i], :t_ys[i]]
            value = values[i, :t_xs[i], :t_ys[i]]
            
            for y in range(t_ys[i]):
                for x in range(max(0, t_xs[i] + y - t_ys[i]), min(t_xs[i], y + 1)):
                    if x == y:
                        v_cur = value[x, y]
                    else:
                        v_cur = value[x, y] + path[x, y-1]
                    if x == 0:
                        if y == 0:
                            v_prev = 0.0
                        else:
                            v_prev = path[x, y-1]
                    else:
                        if y == 0:
                            v_prev = path[x-1, y]
                        else:
                            v_prev = max(path[x-1, y], path[x, y-1])
                    
                    path[x, y] = v_prev + value[x, y]
            
            # Backtrack
            index = t_xs[i] - 1
            for y in range(t_ys[i] - 1, -1, -1):
                path[:, y] = 0.0
                path[index, y] = 1.0
                if index > 0 and (index == 0 or path[index-1, y-1] > path[index, y-1]):
                    index -= 1
        
        return paths


def maximum_path(neg_cent, mask):
    """Find optimal monotonic alignment path using dynamic programming"""
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    mask = mask.data.cpu().numpy()
    
    b, t_x, t_y = neg_cent.shape
    
    # Get actual lengths
    t_xs = mask.sum(axis=1)[:, 0].astype(np.int32)
    t_ys = mask.sum(axis=2)[:, 0].astype(np.int32)
    
    # Initialize paths
    paths = np.zeros((b, t_x, t_y), dtype=np.float32)
    
    if NUMBA_AVAILABLE:
        paths = maximum_path_jit(paths, neg_cent, t_ys, t_xs)
    else:
        # Slow Python fallback
        for i in range(b):
            path = paths[i, :t_xs[i], :t_ys[i]]
            value = neg_cent[i, :t_xs[i], :t_ys[i]]
            
            for y in range(t_ys[i]):
                for x in range(max(0, t_xs[i] + y - t_ys[i]), min(t_xs[i], y + 1)):
                    if x == y:
                        v_cur = value[x, y]
                    else:
                        v_cur = value[x, y] + path[x, y-1]
                    if x == 0:
                        v_prev = 0.0 if y == 0 else path[x, y-1]
                    else:
                        v_prev = path[x-1, y] if y == 0 else max(path[x-1, y], path[x, y-1])
                    
                    path[x, y] = v_prev + value[x, y]
            
            # Backtrack
            index = t_xs[i] - 1
            for y in range(t_ys[i] - 1, -1, -1):
                path[:, y] = 0.0
                path[index, y] = 1.0
                if index > 0 and (index == 0 or path[index-1, y-1] > path[index, y-1]):
                    index -= 1
    
    return torch.from_numpy(paths).to(device=device, dtype=dtype)


def sequence_mask(length, max_length=None):
    """Generate sequence mask"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def duration_loss(logw, logw_, lengths):
    """Compute duration prediction loss"""
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


# ============================================================================
# Data Module
# ============================================================================

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text):
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = None

        return {"x": text, "y": mel, "spk": spk, "filepath": filepath, "x_text": cleaned_text, "durations": durations}

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate, f"Expected {self.sample_rate} Hz, got {sr} Hz"
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }


class TextMelDataModule(LightningDataModule):
    def __init__(
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        self.trainset = TextMelDataset(
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self.validset = TextMelDataset(
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


# ============================================================================
# Lightning Module
# ============================================================================

class MatchaLightningModule(LightningModule):
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        encoder_params,
        decoder_params,
        cfm_params,
        duration_predictor_params,
        data_statistics,
        learning_rate=1e-4,
        prior_loss=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.n_feats = encoder_params.n_feats
        self.prior_loss = prior_loss
        
        # Initialize model
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        
        self.model = MatchaTTS(
            n_vocab=n_vocab,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            cfm_params=cfm_params,
            duration_predictor_params=duration_predictor_params,
        )
        
        # Register mel statistics
        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))
        self.model.mel_mean = self.mel_mean
        self.model.mel_std = self.mel_std

    def forward(self, x, x_lengths, y, y_lengths, spks=None):
        """Forward pass matching original Matcha-TTS"""
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks)
        else:
            spks = None
        
        # Get encoder outputs
        mu_x, logw, x_mask = self.model.encoder(x, x_lengths, spks)
        
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        
        # Monotonic Alignment Search
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const
            
            attn = maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()
        
        # Duration loss
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)
        
        # Align encoder output with mel
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        
        # Flow matching loss (using your fixed compute_loss with masking)
        cfm_loss, _, _, _ = self.model.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=None)
        
        # Prior loss
        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0
        
        return dur_loss, prior_loss, cfm_loss, attn

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        x_lengths = batch["x_lengths"]
        y = batch["y"]
        y_lengths = batch["y_lengths"]
        spks = batch.get("spks", None)
        
        dur_loss, prior_loss, cfm_loss, _ = self(x, x_lengths, y, y_lengths, spks)
        
        loss = dur_loss + prior_loss + cfm_loss
        
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/dur_loss", dur_loss, sync_dist=True)
        self.log("train/prior_loss", prior_loss, sync_dist=True)
        self.log("train/cfm_loss", cfm_loss, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        x_lengths = batch["x_lengths"]
        y = batch["y"]
        y_lengths = batch["y_lengths"]
        spks = batch.get("spks", None)
        
        dur_loss, prior_loss, cfm_loss, _ = self(x, x_lengths, y, y_lengths, spks)
        
        loss = dur_loss + prior_loss + cfm_loss
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/dur_loss", dur_loss, sync_dist=True)
        self.log("val/prior_loss", prior_loss, sync_dist=True)
        self.log("val/cfm_loss", cfm_loss, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# ============================================================================
# Training Functions
# ============================================================================

def create_filelists(data_root):
    """Create train/val filelists with full paths"""
    data_root = Path(data_root)
    metadata_path = data_root / "metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_root}")
    
    # Read metadata and prepend full path to wav files
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Convert to full paths
    processed_lines = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            filename = parts[0]
            wav_path = str(data_root / "wavs" / f"{filename}.wav")
            processed_line = '|'.join([wav_path] + parts[1:]) + '\n'
            processed_lines.append(processed_line)
    
    # Split 95/5
    num_val = int(len(processed_lines) * 0.05)
    train_lines = processed_lines[:-num_val]
    val_lines = processed_lines[-num_val:]
    
    # Write filelists
    train_path = data_root / "train.txt"
    val_path = data_root / "val.txt"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"Created {train_path} with {len(train_lines)} samples")
    print(f"Created {val_path} with {len(val_lines)} samples")
    
    return str(train_path), str(val_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="LJSpeech-1.1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    
    # Create filelists
    train_filelist, val_filelist = create_filelists(args.data_root)
    
    # Model parameters
    encoder_params = SimpleNamespace(
        encoder_type="RoPE Encoder",
        n_feats=80,
        n_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        prenet=True
    )
    
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
        filter_channels_dp=256,
        kernel_size=3,
        p_dropout=0.1
    )
    
    data_statistics = {
        "mel_mean": -5.536622,
        "mel_std": 2.116101
    }
    
    # Initialize DataModule
    print("Initializing TextMelDataModule with bucket batching...")
    datamodule = TextMelDataModule(
        name="ljspeech",
        train_filelist_path=train_filelist,
        valid_filelist_path=val_filelist,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        cleaners=["english_cleaners2"],
        add_blank=True,
        n_spks=1,
        n_fft=1024,
        n_feats=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
        data_statistics=data_statistics,
        seed=42,
        load_durations=False,
    )
    
    # Get vocab size from symbols
    n_vocab = len(symbols)
    print(f"Vocabulary size: {n_vocab}")
    
    # Initialize model
    model = MatchaLightningModule(
        n_vocab=n_vocab,
        n_spks=1,
        spk_emb_dim=64,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        cfm_params=cfm_params,
        duration_predictor_params=duration_predictor_params,
        data_statistics=data_statistics,
        learning_rate=args.lr,
        prior_loss=True,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        every_n_epochs=2,
        save_top_k=3,
        save_last=True,
        filename="matcha-epoch{epoch:02d}-val_loss{val/loss:.2f}"
    )
    
    # Logger
    logger = TensorBoardLogger("lightning_logs", name="matcha_tts")
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
    )
    
    print(f"Starting training...")
    print(f"  Batch size: {args.batch_size} per GPU (Total: {args.batch_size * args.gpus})")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Fully standalone - no external Matcha-TTS imports")
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
