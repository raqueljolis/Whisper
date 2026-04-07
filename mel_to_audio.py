"""
Mel Spectrogram → Audio Converter
Uses a pretrained HiFi-GAN vocoder via SpeechBrain.

Install dependencies:
    pip install speechbrain torchaudio numpy

Supported input formats:
    - .npy files containing mel spectrogram arrays [n_mels, time] or [batch, n_mels, time]
    - .pt  files (PyTorch tensors) with the same shape
"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path


# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Pretrained model options:
#   22,050 Hz (single-speaker, LJ Speech):  "speechbrain/tts-hifigan-ljspeech"
#   16,000 Hz (multi-speaker, LibriTTS):    "speechbrain/tts-hifigan-libritts-16kHz"
#   22,050 Hz (multi-speaker, LibriTTS):    "speechbrain/tts-hifigan-libritts-22050Hz"
MODEL_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
SAMPLE_RATE  = 16000   # must match the model above

INPUT_DIR  = "./mel_inputs"   # folder with your .npy or .pt mel spectrogram files
OUTPUT_DIR = "./audio_outputs" # where .wav files will be saved

# ──────────────────────────────────────────────────────────────────────────────


def load_vocoder(model_source: str):
    """Download (once) and return the pretrained HiFi-GAN vocoder."""
    from speechbrain.inference.vocoders import HIFIGAN
    save_dir = f"pretrained_models/{model_source.split('/')[-1]}"
    print(f"Loading vocoder: {model_source}")
    vocoder = HIFIGAN.from_hparams(source=model_source, savedir=save_dir)
    vocoder.eval()
    return vocoder


def load_mel(path: Path) -> torch.Tensor:
    """
    Load a mel spectrogram from .npy or .pt and return a 3-D tensor
    of shape [1, n_mels, time].
    """
    if path.suffix == ".npy":
        arr = np.load(path)
        mel = torch.from_numpy(arr).float()
    elif path.suffix == ".pt":
        mel = torch.load(path, map_location="cpu").float()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Normalise shape to [batch, n_mels, time]
    if mel.dim() == 2:          # [n_mels, time]
        mel = mel.unsqueeze(0)
    elif mel.dim() == 3:        # already [batch, n_mels, time] — keep as-is
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {mel.shape}")

    return mel


def mel_to_wav(vocoder, mel: torch.Tensor) -> torch.Tensor:
    """
    Convert a mel spectrogram tensor [batch, n_mels, time] to a waveform.
    Returns [batch, samples].
    """
    with torch.no_grad():
        waveforms = vocoder.decode_batch(mel)   # [batch, 1, samples]
    return waveforms.squeeze(1)                 # [batch, samples]


def convert_file(vocoder, input_path: Path, output_path: Path):
    mel = load_mel(input_path)
    wav = mel_to_wav(vocoder, mel)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save each item in the batch (usually just 1)
    for i, audio in enumerate(wav):
        out = output_path.with_stem(output_path.stem + (f"_{i}" if len(wav) > 1 else ""))
        torchaudio.save(str(out), audio.unsqueeze(0), SAMPLE_RATE)
        print(f"  Saved → {out}")


def convert_single(mel_path: str, out_wav_path: str):
    """
    Convenience function: convert one file directly.

    Usage:
        convert_single("my_mel.npy", "output.wav")
    """
    vocoder = load_vocoder(MODEL_SOURCE)
    convert_file(vocoder, Path(mel_path), Path(out_wav_path))


def convert_directory(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    """
    Batch-convert all .npy / .pt files in input_dir → .wav files in output_dir.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    files = list(input_dir.glob("*.npy")) + list(input_dir.glob("*.pt"))
    if not files:
        print(f"No .npy or .pt files found in {input_dir}")
        return

    vocoder = load_vocoder(MODEL_SOURCE)
    print(f"\nConverting {len(files)} file(s)...\n")

    for mel_path in sorted(files):
        out_path = output_dir / (mel_path.stem + ".wav")
        print(f"[{mel_path.name}]")
        convert_file(vocoder, mel_path, out_path)

    print("\nDone.")


# ─── EXAMPLE: convert from a numpy array in memory ───────────────────────────

def convert_numpy_array(mel_array: np.ndarray, out_wav_path: str):
    """
    Convert a numpy mel array directly (no file I/O needed).

    Args:
        mel_array:    numpy array of shape [n_mels, time] or [batch, n_mels, time]
        out_wav_path: path to save the resulting .wav

    Example:
        mel = np.load("sample.npy")          # [80, 300]
        convert_numpy_array(mel, "out.wav")
    """
    vocoder = load_vocoder(MODEL_SOURCE)
    mel = torch.from_numpy(mel_array).float()
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)

    wav = mel_to_wav(vocoder, mel)
    out = Path(out_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), wav[0].unsqueeze(0), SAMPLE_RATE)
    print(f"Saved → {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert mel spectrograms to audio using HiFi-GAN")
    parser.add_argument("--input",  type=str, help="Single .npy/.pt file to convert")
    parser.add_argument("--output", type=str, help="Output .wav path (used with --input)")
    parser.add_argument("--input_dir",  default=INPUT_DIR,  help="Batch input directory")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Batch output directory")
    parser.add_argument("--model", default=MODEL_SOURCE,
                        help="HuggingFace model ID (default: 16kHz LibriTTS HiFi-GAN)")
    args = parser.parse_args()

    MODEL_SOURCE = args.model

    if args.input:
        out = args.output or Path(args.input).with_suffix(".wav").name
        convert_single(args.input, out)
    else:
        convert_directory(args.input_dir, args.output_dir)