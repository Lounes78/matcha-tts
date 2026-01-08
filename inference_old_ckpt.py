import torch
import soundfile as sf
import argparse
import sys
import os
import json
from types import SimpleNamespace
import numpy as np

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MatchaTTS

# -----------------------------------------------------------------------------
# SYMBOLS (MATCHING TRAIN.PY)
# -----------------------------------------------------------------------------
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence_char(text):
    """
    Convert raw text to sequence of IDs (Character-based, NO PHONEMIZER).
    """
    return torch.LongTensor([_symbol_to_id.get(c, 0) for c in text])

# -----------------------------------------------------------------------------
# VOCODER
# -----------------------------------------------------------------------------
def load_vocoder(device):
    # Try to load HiFi-GAN. 
    # Assumes 'hifigan' folder is present and 'generator_v1' checkpoint is available.
    try:
        from hifigan.models import Generator
        
        # We need a config. Checking for config.json in hifigan folder or constructing default
        config = SimpleNamespace(
            resblock=1,
            upsample_rates=[8,8,2,2],
            upsample_kernel_sizes=[16,16,4,4],
            upsample_initial_channel=512,
            resblock_kernel_sizes=[3,7,11],
            resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
            initial_channel=128,  # Check this! Default HiFiGAN V1 is usually 128 or 512? 
                                  # Original HiFiGAN config_v1.json: input_channels=128? 
                                  # Wait, Matcha-TTS typically uses 80-band mel.
                                  # The Generator constructor takes `h` (params).
        )
        
        # Actually simplest way is to load the state dict and see.
        # But we need the class.
        
        # Let's inspect hifigan/config.json if it exists.
        config_path = "hifigan/config.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                data = json.load(f)
            config = SimpleNamespace(**data)
        else:
             print("Warning: hifigan/config.json not found. Using default V1 params.")
             # Default V1 params
             config = SimpleNamespace(
                 resblock="1",
                 upsample_rates=[8,8,2,2],
                 upsample_kernel_sizes=[16,16,4,4],
                 upsample_initial_channel=512,
                 resblock_kernel_sizes=[3,7,11],
                 resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                 model_in_dim=80
             )

        vocoder = Generator(config)
        vocoder.load_state_dict(torch.load("generator_v1", map_location=device)['generator'])
        vocoder.to(device)
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder
        
    except Exception as e:
        print(f"Failed to load vocoder: {e}")
        return None

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--text", type=str, default="Light of the seven.", help="Text to speak")
    parser.add_argument("--output", type=str, default="old_ckpt_output.wav", help="Output wav file")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model Params (Same defaults as train.py)
    encoder_params = SimpleNamespace(
        encoder_type="roformer", n_feats=80, n_channels=192, filter_channels=768,
        n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, prenet=True
    )
    decoder_params = SimpleNamespace(
        channels=(256, 256), num_res_blocks=2, num_transformer_blocks=1,
        num_heads=2, time_emb_dim=256, time_mlp_dim=512, dropout=0.05, ffn_mult=4
    )
    cfm_params = {"solver": "euler", "sigma_min": 1e-4}
    duration_predictor_params = SimpleNamespace(
        filter_channels_dp=256, kernel_size=3, p_dropout=0.1
    )
    params = {
        "n_vocab": len(symbols),
        "n_spks": 1,
        "spk_emb_dim": 64,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "cfm_params": cfm_params,
        "duration_predictor_params": duration_predictor_params
    }
    
    print("Initializing Model...")
    model = MatchaTTS(**params).to(device)
    
    # 2. Load Checkpoint (Handling 'model.' prefix)
    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        print("Trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    
    # 3. Load Vocoder
    print("Loading Vocoder (HiFi-GAN)...")
    vocoder = load_vocoder(device)
    if vocoder is None:
        print("Error: Could not load vocoder. Ensure 'generator_v1' is present.")
        sys.exit(1)
        
    # 4. Inference
    print(f"Synthesizing: '{args.text}'")
    x = text_to_sequence_char(args.text).unsqueeze(0).to(device)
    x_lengths = torch.tensor([x.shape[1]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Matcha Flow Matching
        # Using model.synthesize logic which is implemented in MatchaTTS class now (I saw it in model.py)
        
        # But wait, model.py in the previous read call has `synthesize` method at the end?
        # Yes, lines 1282+
        # But it requires `fix_len_compatibility` and `generate_path` which might not be imported in model.py context?
        # Let's check imports in model.py or if we need to call model.synthesize directly.
        
        # If model.synthesize is available, use it!
        if hasattr(model, 'synthesize'):
            print("Using model.synthesize()...")
            # We need to set mel stats if they were used.
            # Default Matcha uses stats from LJSpeech.
            # If our training didn't use normalization, we should set mean=0, std=1.
            # Our train.py didn't use normalization on Mels (it generated them raw).
            # So we should ensure model uses mean=0, std=1.
            # Updated based on calc_stats.py results for standard LJSpeech/HiFiGAN config
            model.mel_mean = torch.tensor(-5.5044, device=device)
            model.mel_std = torch.tensor(2.0493, device=device)
            
            mel, mel_lengths = model.synthesize(x, x_lengths, n_timesteps=50, temperature=0.667, spks=None, length_scale=1.0)
            
        else:
            print("Error: model.synthesize not found. Implementing manual inference loop...")
            # ... (Fallback manual loop if needed)
            sys.exit(1)

        # 5. Vocode
        print("Vocoding...")
        audio = vocoder(mel)
        audio = audio.squeeze().cpu().numpy()
        
        print(f"Saving to {args.output}...")
        sf.write(args.output, audio, 22050)
        print("Done.")

if __name__ == "__main__":
    main()
