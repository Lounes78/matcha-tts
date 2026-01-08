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

def maximum_path(value, mask):
    """
    Monotonic Alignment Search (PyTorch Implementation)
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    device = value.device
    dtype = value.dtype
    value = value * mask
    
    b, t_x, t_y = value.shape
    direction = torch.zeros(b, t_x, t_y, dtype=torch.long, device=device)
    v = torch.zeros(b, t_x, t_y, dtype=dtype, device=device)
    
    # Initialize
    # Dynamic programming buffer
    neg_inf = -1e9
    
    # (Simplified Viterbi for Monotonic Alignment)
    # We want to find path from (0,0) to (t_x-1, t_y-1)
    
    v = torch.full_like(value, neg_inf)
    v[:, 0, 0] = value[:, 0, 0]
    
    # Forward pass
    for j in range(1, t_y):
        for i in range(t_x):
            src_v = v[:, i, j-1]
            if i > 0:
                src_prev = v[:, i-1, j-1]
                # Compare src_v and src_prev
                # We need to take max and store direction
                # direction=0 : from same i (stay)
                # direction=1 : from i-1 (move)
                stack = torch.stack([src_v, src_prev], dim=-1)
                best_val, best_idx = torch.max(stack, dim=-1)
                v[:, i, j] = best_val + value[:, i, j]
                direction[:, i, j] = best_idx
            else:
                v[:, i, j] = src_v + value[:, i, j]
                direction[:, i, j] = 0 # Can only come from same i
                
    # Backward pass
    text_len = mask.sum(dim=1)[:,0].long()
    mel_len = mask.sum(dim=2)[:,0].long()
    
    path = torch.zeros(b, t_x, t_y, dtype=dtype, device=device)
    
    for k in range(b):
        tx = text_len[k] - 1
        ty = mel_len[k] - 1
        
        while ty >= 0 and tx >= 0:
            path[k, tx, ty] = 1
            if ty == 0:
                break
            if direction[k, tx, ty] == 1:
                tx = max(0, tx - 1)
            ty = max(0, ty - 1)
            
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

    def forward(self, x, x_lengths, y, y_lengths, spks=None):
        return self.model(x, x_lengths, y, y_lengths, spks)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        x, x_lengths = batch['text'], batch['text_lengths']
        spks = batch.get('spks', None)
        
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
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(y.device) # [b, 1, t_y]
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) # [b, 1, t_x, t_y]
            
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
        w_path = path.sum(dim=2) # [b, t_x]
        
        # Duration Loss
        # logw is predicted log duration.
        loss_duration = F.mse_loss(logw, torch.log(w_path.unsqueeze(1) + 1e-8), reduction='none')
        loss_duration = (loss_duration * x_mask).sum() / x_mask.sum()
        
        # 4. Upsample mu using path for CFM
        # path: [b, t_x, t_y]
        # mu: [b, c, t_x]
        # mu_y = mu * path
        mu_y = torch.matmul(mu, path) # [b, c, t_x] * [b, t_x, t_y] -> [b, c, t_y]
        
        # 5. CFM Loss
        # compute_loss calls estimator with mu_y (aligned)
        
        # Handle Speaker Embedding
        if self.model.n_spks > 1 and spks is not None:
            spks = self.model.spk_emb(spks)
        else:
            spks = None
            
        cfm_loss, _, _, _ = self.model.decoder.compute_loss(y, y_mask, mu_y, spks, cond=None)
        
        loss = cfm_loss + loss_duration
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/cfm_loss", cfm_loss)
        self.log("train/dur_loss", loss_duration)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.8, 0.99))
        return optimizer

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class LJSpeechDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.wav_dir = self.data_root / "wavs"
        self.metadata_path = self.data_root / "metadata.csv"

        # Init Phonemizer
        try:
            import phonemizer
            self.global_phonemizer = phonemizer.backend.EspeakBackend(
                language='en-us', preserve_punctuation=True, with_stress=True
            )
            print("Phonemizer loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load phonemizer: {e}")
            self.global_phonemizer = None
        
        if not self.metadata_path.exists():
             raise FileNotFoundError(f"Metadata not found at {self.metadata_path}. Please download LJSpeech-1.1 and extract it.")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = [line.strip().split('|') for line in f]
            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        file_id = item[0]
        # Index 2 is cleaner text, Index 1 is original
        text_str = item[2] if len(item) > 2 else item[1]
        
        # Phonemize on-the-fly
        if self.global_phonemizer:
            try:
                phonemized_text = self.global_phonemizer.phonemize([text_str], strip=True)[0]
                text_ints = [ _symbol_to_id.get(c, 0) for c in phonemized_text ]
            except Exception as e:
                print(f"Error: Phonemization failed for '{text_str}': {e}. Retrying with next sample...")
                return self.__getitem__((idx + 1) % len(self))
        else:
             text_ints = [ _symbol_to_id.get(c, 0) for c in text_str ]

        text = torch.LongTensor(text_ints)
        
        wav_path = self.wav_dir / f"{file_id}.wav"
        if not wav_path.exists():
             print(f"Warning: {wav_path} not found")
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
    encoder_params = SimpleNamespace(
        encoder_type="roformer", n_feats=80, n_channels=192, filter_channels=768,
        n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, prenet=True
    )
    # Decoder Params:
    # channels=[256, 256], n_blocks=1 (transformer depth), num_mid_blocks=2, num_heads=2
    # dropout=0.05 (matches config/model/decoder/default.yaml)
    decoder_params = SimpleNamespace(
        channels=(256, 256), num_res_blocks=2, num_transformer_blocks=1,
        num_heads=2, time_emb_dim=256, time_mlp_dim=512, dropout=0.05, ffn_mult=4
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
    args = parser.parse_args()

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
    train_dataset = LJSpeechDataset(args.data_root) 
    
    # Auto-adjust workers: 4 GPUs -> likely powerful CPU. Use 16 workers.
    num_workers = 16 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Trainer
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger("lightning_logs", name="matcha_tts")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="train/loss", 
        mode="min", 
        every_n_epochs=3,
        save_top_k=-1, # Keep all checkpoints saved every 3 epochs
        filename="matcha-epoch{epoch:02d}-loss{train/loss:.2f}"
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=5
    )
    
    print(f"Starting training on {args.gpus} GPUs with precision {args.precision}...")
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
