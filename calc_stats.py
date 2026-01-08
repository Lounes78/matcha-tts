import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import sys

# Try to use the same mel function as training
# Copied from hifigan/meldataset.py
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    y = torch.from_numpy(y).unsqueeze(0)
    mel_basis = {}
    hann_window = {}
    
    mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis, spec)
    # Log compression
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec

def main():
    root = Path("LJSpeech-1.1/wavs")
    if not root.exists():
        print(f"Error: {root} not found")
        return

    wav_files = list(root.glob("*.wav"))[:100] # Use 100 files
    print(f"Calculating stats on {len(wav_files)} files...")

    mel_sum = 0
    mel_sq_sum = 0
    total_frames = 0
    
    # HiFiGAN Defaults
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    sampling_rate = 22050
    num_mels = 80
    fmin = 0
    fmax = 8000

    for wf in tqdm(wav_files):
        y, sr = librosa.load(wf, sr=sampling_rate)
        # Trim silence (optional, but good for stats)
        y, _ = librosa.effects.trim(y)
        
        mel = mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        # mel: [1, 80, T]
        
        mel_sum += torch.sum(mel)
        mel_sq_sum += torch.sum(mel ** 2)
        total_frames += mel.shape[2] * mel.shape[1] # Total elements

    # Global Mean/Std (Scalar)
    # If we want per-channel, we'd adjust dimensions
    mean = mel_sum / total_frames
    std = torch.sqrt((mel_sq_sum / total_frames) - mean ** 2)

    print(f"Global Mean: {mean.item()}")
    print(f"Global Std: {std.item()}")

if __name__ == "__main__":
    main()
