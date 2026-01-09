import torch
import numpy as np
import os
import sys
from types import SimpleNamespace
import soundfile as sf

from model import MatchaTTS

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CHECKPOINT_PATH = "matcha_ljspeech.ckpt"
CHECKPOINT_PATH = "/home/lounes/tts/matcha-tts/lightning_logs/matcha_tts/version_29/checkpoints/last.ckpt"
VOCODER_URL = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1"
VOCODER_PATH = "generator_v1"

# ----------------------------------------------------------------------------------------------------------------------
# 1. TEXT PREPARATION (Phonemizer & Symbols)
# ----------------------------------------------------------------------------------------------------------------------
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def get_phonemized_text(text):
    # return "hˈeɪ, ðˈɪs ɪz ə tˈɛst wɪð ˈaʊɚ stˈændəˌloʊn fˈaɪl ˌɪmplɪmɛntˈeɪʃən !"
    try:
        import phonemizer
        # Use espeak-ng backend
        global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True
        )
        return global_phonemizer.phonemize([text], strip=True)[0]
    except ImportError:
        print("Phonemizer not installed or espeak-ng missing. Using default phonemes.")
        # Fallback for testing without phonemizer
        return "heɪ, aɪ æm mætʃə ə tɛkst tu spitʃ..."

def phonemes_to_sequence(phonemized_text):
    sequence = []
    for symbol in phonemized_text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        else:
            print(f"Warning: Symbol '{symbol}' not found in vocabulary, skipping.")
    return sequence

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

# ----------------------------------------------------------------------------------------------------------------------
# 2. MODEL LOADING (The Custom Part)
# ----------------------------------------------------------------------------------------------------------------------
def load_custom_matcha(checkpoint_path, device):
    print(f"Loading custom model from {checkpoint_path}...")
    
    encoder_params = SimpleNamespace(
        encoder_type="RoPE Encoder", n_feats=80, n_channels=192, filter_channels=768,
        n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, prenet=True
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
        filter_channels_dp=256, kernel_size=3, p_dropout=0.1
    )
    
    # B. Instantiate your local model class
    model = MatchaTTS(
        n_vocab=len(symbols),
        n_spks=1,
        spk_emb_dim=64,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        cfm_params=cfm_params,
        duration_predictor_params=duration_predictor_params
    ).to(device)

    # C. Load Weights manually
    # We load the 'state_dict' from the lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if state_dict is inside a key (common in Lightning)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # The checkpoint keys start with "model." because LightningModule wrapper has self.model
    # But we are loading into the inner model directly.
    # We need to strip "model." prefix from keys.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v # Remove "model."
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    if "mel_mean" in state_dict:
        print(f"Found mel_mean in checkpoint: {state_dict['mel_mean']}")
        print(f"Found mel_std in checkpoint: {state_dict['mel_std']}")
    else:
        print("WARNING: mel_mean NOT found in checkpoint!")

    # Attempt to load
    try:
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print("Tip: If keys don't match, your model.py structure might differ from the official repo.")
        sys.exit(1)

    model.eval()
    return model

# ----------------------------------------------------------------------------------------------------------------------
# 3. VOCODER LOADING (HiFiGAN)
# ----------------------------------------------------------------------------------------------------------------------
def load_vocoder(device):
    import urllib.request
    from hifigan.models import Generator as HiFiGAN
    from hifigan.config import v1
    from hifigan.env import AttrDict

    if not os.path.exists(VOCODER_PATH):
        print("Downloading Vocoder...")
        urllib.request.urlretrieve(VOCODER_URL, VOCODER_PATH)

    h = AttrDict(v1)
    vocoder = HiFiGAN(h).to(device)
    state = torch.load(VOCODER_PATH, map_location=device)
    vocoder.load_state_dict(state["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder

# ----------------------------------------------------------------------------------------------------------------------
# 4. MAIN INFERENCE LOOP
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # A. Text Input
    text = "Hello! I am running on your custom model file."
    text = "They printed very few books in this type, three only; but in their very first books in Rome, beginning with the year 1468,|They printed very few books in this type, three only; but in their very first books in Rome, beginning with the year fourteen sixty-eight,"
    print(f"Input Text: {text}")

    # B. Process Text
    phonemes = get_phonemized_text(text)
    print(f"Phonemes: {phonemes}")
    
    token_ids = phonemes_to_sequence(phonemes)
    interspersed_ids = intersperse(token_ids, 0)
    
    x = torch.tensor(interspersed_ids, dtype=torch.long, device=DEVICE)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=DEVICE)

    # C. Load Models
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint {CHECKPOINT_PATH} not found. Please download it first.")
        return

    model = load_custom_matcha(CHECKPOINT_PATH, DEVICE)
    
    # D. Synthesize Mel Spectrogram
    # Note: We use model.synthesize (with z) as defined in your model.py
    print("Generating Mel Spectrogram...")
    with torch.inference_mode():
        output = model.synthesize(
            x, 
            x_lengths, 
            n_timesteps=10, 
            temperature=0.667, 
            length_scale=1.0
        )
        mel = output[0] # output is (mel, lengths) tuple in your model.py
        
        print(f"Mel Shape: {mel.shape}")
        print(f"Mel Min: {mel.min().item():.4f}, Max: {mel.max().item():.4f}, Mean: {mel.mean().item():.4f}")
        
    # E. Generate Audio with Vocoder
    print("Generating Audio...")
    vocoder = load_vocoder(DEVICE)
    with torch.inference_mode():
        audio = vocoder(mel).clamp(-1, 1).cpu().squeeze()
    
    # F. Save
    sf.write("infer_output.wav", audio.numpy(), 22050)
    print("Done! Saved to infer_output.wav")
    
    # G. Plot Attention (Validation)
    import matplotlib.pyplot as plt
    attn = output[2].squeeze().cpu().numpy() # output[2] is attn from synthesize
    plt.figure(figsize=(10, 4))
    plt.imshow(attn, origin='lower', aspect='auto')
    plt.colorbar()
    plt.title("Alignment (Attention)")
    plt.xlabel("Mel Frames")
    plt.ylabel("Text Tokens")
    plt.savefig("alignment.png")
    print("Saved alignment plot to alignment.png")

if __name__ == "__main__":
    main()