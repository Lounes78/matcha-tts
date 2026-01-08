import argparse
import os
from pathlib import Path
from tqdm import tqdm
from phonemizer.backend import EspeakBackend

def main():
    parser = argparse.ArgumentParser(description="Pre-phonemize LJSpeech metadata")
    parser.add_argument("--root", type=str, default="LJSpeech-1.1", help="Path to LJSpeech root")
    parser.add_argument("--output", type=str, default="metadata_phonemes.csv", help="Output metadata filename")
    parser.add_argument("--lang", type=str, default="en-us", help="Phonemizer language")
    args = parser.parse_args()

    root_dir = Path(args.root)
    metadata_path = root_dir / "metadata.csv"
    output_path = root_dir / args.output

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        # Try finding it in the parent or subfolders just in case
        print(f"Current CWD: {os.getcwd()}")
        return

    print(f"Initializing phonemizer backend ({args.lang})...")
    backend = EspeakBackend(args.lang, preserve_punctuation=True, with_stress=True)

    print(f"Reading {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Phonemizing {len(lines)} lines...")
    
    new_lines = []
    
    # Check separator
    sep = '|'
    
    for line in tqdm(lines):
        parts = line.strip().split(sep)
        file_id = parts[0]
        
        # LJSpeech format: ID|Transcription|Normalized Transcription
        # We prefer Normalized (index 2) if available, else Transcription (index 1)
        if len(parts) >= 3:
            text = parts[2]
        else:
            text = parts[1]
            
        try:
            # Phonemize
            # strip=True removes start/end whitespace
            phonemes = backend.phonemize([text], strip=True)[0]
        except Exception as e:
            print(f"Failed to phonemize {file_id}: {e}")
            # Fallback to text
            phonemes = text

        # Output format: ID|Phonemes
        new_lines.append(f"{file_id}|{phonemes}\n")

    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    print("Done!")

if __name__ == "__main__":
    main()
