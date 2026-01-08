import torch
import re
from phonemizer import phonemize

# Define the symbol to ID mapping (phoneme alphabet)
_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Combine all symbols
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Create symbol to ID mapping
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}







def process_text(text, device='cpu'):
    """
    Complete pipeline: text -> phonemes -> sequence -> tensor
    
    Args:
        text: input text string
        device: torch device ('cpu' or 'cuda')
    
    Returns:
        dict with processed data
    """
    
    # Step 1: Clean text and convert to phonemes
    phonemes = english_cleaners(text)
    
    # Step 2: Convert phonemes to sequence of IDs
    sequence = text_to_sequence(phonemes)
    
    # Step 3: Intersperse with blanks (0s)
    sequence_with_blanks = intersperse(sequence, 0)    
    
    # Step 4: Convert to PyTorch tensor
    x = torch.tensor(sequence_with_blanks, dtype=torch.long, device=device)
    
    # Step 5: Add batch dimension
    x = x.unsqueeze(0)  # Same as [None] - adds dimension at front
    
    # Calculate length
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    print("Processing complete!")
    return {
        'x_orig': text,              # Original text
        'x_phones': phonemes,         # Phonetic representation
        'x': x,                       # Tensor with batch dimension
        'x_lengths': x_lengths,       # Length tensor
        'sequence': sequence          # Raw sequence (for debugging)
    }










def english_cleaners(text):
    """
    Pipeline for English text processing:
    1. Convert to ASCII
    2. Lowercase
    3. Expand abbreviations
    4. Convert to phonemes
    5. Clean up
    """
    # Step 1: Convert to ASCII
    text = convert_to_ascii(text)    
    # Step 2: Lowercase
    text = lowercase(text)    
    # Step 3: Expand abbreviations
    text = expand_abbreviations(text)    
    # Step 4: Convert to phonemes using phonemizer
    phonemes = phonemize(
        text,
        language='en-us',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=True
    )
    
    # Step 5: Remove brackets
    phonemes = remove_brackets(phonemes)    
    # Step 6: Collapse whitespace
    phonemes = collapse_whitespace(phonemes)    
    return phonemes


def text_to_sequence(text):
    """
    Convert cleaned phonetic text to sequence of IDs
    
    Args:
        text: phonetic text string
    
    Returns:
        sequence: list of integer IDs
    """
    sequence = []
    
    # Convert each character/phoneme to its ID
    for symbol in text:
        if symbol in _symbol_to_id:
            symbol_id = _symbol_to_id[symbol]
            sequence.append(symbol_id)
        else:
            # If symbol not in vocabulary, skip it or use a default
            print(f"Warning: Symbol '{symbol}' not in vocabulary, skipping")
    
    return sequence


def intersperse(sequence, item):
    """
    Insert item between every element in sequence
    
    Args:
        sequence: list of elements
        item: item to insert between elements
    
    Returns:
        new list with item interspersed
    
    Example:
        intersperse([1, 2, 3], 0) -> [1, 0, 2, 0, 3]
    """
    result = []
    for i, element in enumerate(sequence):
        result.append(element)
        if i < len(sequence) - 1:  # Don't add after last element
            result.append(item)
    return result





 
def convert_to_ascii(text):
    """Convert text to ASCII, removing accents"""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def lowercase(text):
    """Convert text to lowercase"""
    return text.lower()


def expand_abbreviations(text):
    """Expand common abbreviations"""
    abbreviations = {
        'mr.': 'mister',
        'mrs.': 'misess',
        'dr.': 'doctor',
        'st.': 'saint',
        'co.': 'company',
        'jr.': 'junior',
        'sr.': 'senior',
        'etc.': 'et cetera',
        'vs.': 'versus',
        'ltd.': 'limited',
    }
    
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
    
    return text


def remove_brackets(text):
    """Remove any brackets from text"""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    return text


def collapse_whitespace(text):
    """Replace multiple spaces with single space"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


