# import phonemizer


# # 1. we phonemize
# phonemizer_backend = phonemizer.backend.EspeakBackend(language="en-us", with_stress=True)

# text = 'Hey, I am Matcha a text to speech using Optimal transport conditional flow matching.'
# phonemized_text = phonemizer_backend.phonemize([text.lower()], strip=True)[0]

# print(f"Original: {text}")
# print(f"Phonemized: {phonemized_text}")

# ----------------------------------------------------------------------------------------------------------------------

# 2. Phonemes Token IDs
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def phonemes_to_sequence(phonemized_text):
    sequence = [_symbol_to_id[symbol] for symbol in phonemized_text]
    return sequence


phonemized_text = "heɪ, aɪ æm mætʃə ə tɛkst tu spitʃ..."
# phonemized_text = "heɪ, aɪ æm jɔr tɔkiŋ əˈlaɪdʒən. aɪ kæn tɜrn jɔr tɛkst ˈɪntu saʊndz laɪk məˈdʒɪk. ˈsʌmtaɪmz aɪ siŋ, ˈsʌmtaɪmz aɪ ˈmʌmbəl, ænd ˈsʌmtaɪmz aɪ stɑrt ˈspiːdˌrʌnɪŋ sɛnˌtɛnsɪz fɔr no riːzən. soʊ ændʒɔɪ, bɪˈkɔz aɪm ˈprɑbəbli ˈwɔtʃɪŋ ju frəm ðə ˈklaʊdz raɪt naʊ..."

token_ids = phonemes_to_sequence(phonemized_text)
print(f"Token IDs: {token_ids}")
# [50, 47, 102, 3, 16, 43, .....


# ----------------------------------------------------------------------------------------------------------------------

import torch
## 3. Prepare model input (CORRECTED)
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def prepare_model_input(token_ids, device="cpu"):
    interspersed = intersperse(token_ids, 0)
    
    x = torch.tensor(interspersed, dtype=torch.long, device=device)[None]  
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return x, x_lengths

x, x_lengths = prepare_model_input(token_ids)
print(f"Model input shape: {x.shape}")

print(f"Original: {token_ids}")
print(f"Interspersed: {intersperse(token_ids, 0)}")

#  [0, 50, 0, 47, 0, 102, 0, 3, 0, 16, 0, 43, 0, .....



# # ----------------------------------------------------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from matcha.models.matcha_tts import MatchaTTS

#  Invoke-WebRequest -Uri "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt" -OutFile "matcha_ljspeech.ckpt"

# 4. Load the pre trained Matcha model
def load_matcha_model(device="cpu"):
    checkpoint_path = "matcha_ljspeech.ckpt"
    
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_matcha_model(device)

# 5. Gnerate the mel spectrogram
def generate_mel(model, x, x_lengths, device="cpu"):
    with torch.inference_mode():
        output = model.synthesise(
            x.to(device), 
            x_lengths.to(device),
            n_timesteps=10,        
            temperature=0.667,     
            length_scale=1.0     
        )
        return output["mel"]  

mel_spectrogram = generate_mel(model, x, x_lengths, device)
print(f"Generated mel shape: {mel_spectrogram.shape}")

# 7. Load HiFiGAN vocoder and generate waveform
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.hifigan.config import v1
from matcha.hifigan.env import AttrDict
import urllib.request

def load_hifigan_vocoder(device="cpu"):
    vocoder_url = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1"
    vocoder_path = "generator_v1"
    
    if not os.path.exists(vocoder_path):
        print(f"Downloading {vocoder_path}...")
        urllib.request.urlretrieve(vocoder_url, vocoder_path)
        print(f"Downloaded {vocoder_path}")
    
    h = AttrDict(v1)
    vocoder = HiFiGAN(h).to(device)
    vocoder.load_state_dict(torch.load(vocoder_path, map_location=device)["generator"])
    vocoder.eval()
    return vocoder

# 8. Generate audio
vocoder = load_hifigan_vocoder(device)
mel_for_vocoder = mel_spectrogram.clone()

with torch.no_grad():  
    audio = vocoder(mel_for_vocoder).clamp(-1, 1).cpu().squeeze()

import soundfile as sf
sf.write("output.wav", audio.numpy(), 22050)
print("Complete TTS pipeline finished!")
print("Audio saved as output.wav")
print(f"Final audio shape: {audio.shape} samples")
print(f"Audio duration: {audio.shape[0] / 22050:.2f} seconds")