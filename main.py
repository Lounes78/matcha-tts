import torch
import numpy as np
import os
import sys
from types import SimpleNamespace
import soundfile as sf

# Add valid path to find 'matcha' package in parent directory

# 1. Import your local model
# Ensure model.py is in the same directory
from model import MatchaTTS

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "matcha_ljspeech.ckpt"
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
    
    # A. Define Hyperparameters for LJSpeech (Matcha-TTS Standard)
    # These must match the config used to train the checkpoint
    # Updated to match standard LJSpeech config: channels=[256, 256], time_mlp_dim=1024?? No wait, let's check validation.
    # The error said: size mismatch for decoder.estimator.time_mlp.linear_1.weight: copying a param with shape torch.Size([1024, 160]) from checkpoint.
    # So time_mlp_dim is 1024 (output of linear_1), and input is 160 (time_enc_dim=160?? wait).
    # Linear 1 is in_features -> time_embed_dim.
    # If Linear 1 weight is [1024, 160], then out=1024, in=160.
    # If input is 160, and SinusoidalPosEmb(in_channels) is used, then the Decoder in_channels must be 160.
    # If out is 1024, then time_embed_dim (inner) must be 1024?
    # Original Decoder code: time_embed_dim = channels[0] * 4.
    # If channels[0] is 256, then 256 * 4 = 1024. Matches!
    
    encoder_params = SimpleNamespace(
        encoder_type="roformer", n_feats=80, n_channels=192, filter_channels=768,
        n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, prenet=True
    )
    decoder_params = SimpleNamespace(
        channels=(256, 256), num_res_blocks=2, num_transformer_blocks=1,
        num_heads=2, time_emb_dim=256, time_mlp_dim=512, dropout=0.0, ffn_mult=4
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

    # DEBUG: Check for mel_mean/std in state_dict
    if "mel_mean" in state_dict:
        print(f"Found mel_mean in checkpoint: shape {state_dict['mel_mean'].shape}")
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

    # E. Generate Audio with Vocoder
    print("Generating Audio...")
    vocoder = load_vocoder(DEVICE)
    with torch.inference_mode():
        audio = vocoder(mel).clamp(-1, 1).cpu().squeeze()
    
    # F. Save
    sf.write("infer_output.wav", audio.numpy(), 22050)
    print("Done! Saved to infer_output.wav")

if __name__ == "__main__":
    main()



# # import phonemizer


# # # 1. we phonemize
# # phonemizer_backend = phonemizer.backend.EspeakBackend(language="en-us", with_stress=True)

# # text = 'Hey, I am Matcha a text to speech using Optimal transport conditional flow matching.'
# # phonemized_text = phonemizer_backend.phonemize([text.lower()], strip=True)[0]

# # print(f"Original: {text}")
# # print(f"Phonemized: {phonemized_text}")

# # ----------------------------------------------------------------------------------------------------------------------

# # 2. Phonemes Token IDs
# _pad = "_"
# _punctuation = ';:,.!?¡¿—…"«»"" '
# _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# _symbol_to_id = {s: i for i, s in enumerate(symbols)}

# def phonemes_to_sequence(phonemized_text):
#     sequence = [_symbol_to_id[symbol] for symbol in phonemized_text]
#     return sequence


# phonemized_text = "heɪ, aɪ æm mætʃə ə tɛkst tu spitʃ..."
# # phonemized_text = "heɪ, aɪ æm jɔr tɔkiŋ əˈlaɪdʒən. aɪ kæn tɜrn jɔr tɛkst ˈɪntu saʊndz laɪk məˈdʒɪk. ˈsʌmtaɪmz aɪ siŋ, ˈsʌmtaɪmz aɪ ˈmʌmbəl, ænd ˈsʌmtaɪmz aɪ stɑrt ˈspiːdˌrʌnɪŋ sɛnˌtɛnsɪz fɔr no riːzən. soʊ ændʒɔɪ, bɪˈkɔz aɪm ˈprɑbəbli ˈwɔtʃɪŋ ju frəm ðə ˈklaʊdz raɪt naʊ..."

# token_ids = phonemes_to_sequence(phonemized_text)
# print(f"Token IDs: {token_ids}")
# # [50, 47, 102, 3, 16, 43, .....


# # ----------------------------------------------------------------------------------------------------------------------

# import torch
# ## 3. Prepare model input (CORRECTED)
# def intersperse(lst, item):
#     result = [item] * (len(lst) * 2 + 1)
#     result[1::2] = lst
#     return result

# def prepare_model_input(token_ids, device="cpu"):
#     interspersed = intersperse(token_ids, 0)
    
#     x = torch.tensor(interspersed, dtype=torch.long, device=device)[None]  
#     x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
#     return x, x_lengths

# x, x_lengths = prepare_model_input(token_ids)
# print(f"Model input shape: {x.shape}")

# print(f"Original: {token_ids}")
# print(f"Interspersed: {intersperse(token_ids, 0)}")

# #  [0, 50, 0, 47, 0, 102, 0, 3, 0, 16, 0, 43, 0, .....



# # # ----------------------------------------------------------------------------------------------------------------------

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from matcha.models.matcha_tts import MatchaTTS

# #  Invoke-WebRequest -Uri "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt" -OutFile "matcha_ljspeech.ckpt"

# # 4. Load the pre trained Matcha model
# def load_matcha_model(device="cpu"):
#     checkpoint_path = "matcha_ljspeech.ckpt"
    
#     model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
#     model.eval()
#     return model

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = load_matcha_model(device)

# # 5. Gnerate the mel spectrogram
# def generate_mel(model, x, x_lengths, device="cpu"):
#     with torch.inference_mode():
#         output = model.synthesise(
#             x.to(device), 
#             x_lengths.to(device),
#             n_timesteps=10,        
#             temperature=0.667,     
#             length_scale=1.0     
#         )
#         return output["mel"]  

# mel_spectrogram = generate_mel(model, x, x_lengths, device)
# print(f"Generated mel shape: {mel_spectrogram.shape}")

# # 7. Load HiFiGAN vocoder and generate waveform
# from matcha.hifigan.models import Generator as HiFiGAN
# from matcha.hifigan.config import v1
# from matcha.hifigan.env import AttrDict
# import urllib.request

# def load_hifigan_vocoder(device="cpu"):
#     vocoder_url = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1"
#     vocoder_path = "generator_v1"
    
#     if not os.path.exists(vocoder_path):
#         print(f"Downloading {vocoder_path}...")
#         urllib.request.urlretrieve(vocoder_url, vocoder_path)
#         print(f"Downloaded {vocoder_path}")
    
#     h = AttrDict(v1)
#     vocoder = HiFiGAN(h).to(device)
#     vocoder.load_state_dict(torch.load(vocoder_path, map_location=device)["generator"])
#     vocoder.eval()
#     return vocoder

# # 8. Generate audio
# vocoder = load_hifigan_vocoder(device)
# mel_for_vocoder = mel_spectrogram.clone()

# with torch.no_grad():  
#     audio = vocoder(mel_for_vocoder).clamp(-1, 1).cpu().squeeze()

# import soundfile as sf
# sf.write("output.wav", audio.numpy(), 22050)
# print("Complete TTS pipeline finished!")
# print("Audio saved as output.wav")
# print(f"Final audio shape: {audio.shape} samples")
# print(f"Audio duration: {audio.shape[0] / 22050:.2f} seconds")