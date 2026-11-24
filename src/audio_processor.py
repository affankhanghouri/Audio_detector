"""Decode audio, validate the first ten seconds, and reproduce training input."""

from pathlib import Path

import librosa
import numpy as np
import torch

from .config import (
    DURATION, EXPECTED_TENSOR_SHAPE, HOP_LENGTH, MIN_CENTERED_RMS,
    N_FFT, N_MELS, SAMPLE_RATE, SUPPORTED_AUDIO_SUFFIXES,
)


def preprocess_audio(signal):
    """Match the Kaggle preprocessing exactly, without resizing or extra scaling."""
    signal = signal - signal.mean()

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=5000,
        power=2.0,
    )
    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
        amin=1e-20,
        top_db=80,
    )
    normalized = np.clip((log_mel + 80) / 80, 0, 1)
    return normalized.astype(np.float32)


def load_audio(audio_path):
    """Return ten seconds of finite, non-silent mono audio at 10 kHz."""
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError("Unsupported audio format. Please provide a WAV or MP3 file.")
    try:
        # Check duration at the native rate before resampling can round it upward.
        # Limit decoding to the first ten seconds for long recordings.
        signal, native_rate = librosa.load(
            path, sr=None, mono=True, duration=DURATION, dtype=np.float32,
        )
    except Exception as exc:
        raise ValueError(f"Cannot decode audio '{path.name}': {exc}") from exc

    if signal.size < native_rate * DURATION:
        raise ValueError(
            f"Audio must be at least {DURATION} seconds long; "
            f"decoded {signal.size / native_rate:.6f} seconds."
        )
    if not np.isfinite(signal).all():
        raise ValueError("Audio contains NaN or infinite values.")
    if native_rate != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=native_rate, target_sr=SAMPLE_RATE)
    signal = signal[:SAMPLE_RATE * DURATION]
    if not np.isfinite(signal).all():
        raise ValueError("Resampling produced NaN or infinite values.")
    centered = signal.astype(np.float64) - signal.mean(dtype=np.float64)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    if rms <= MIN_CENTERED_RMS:
        raise ValueError(
            f"Audio is silent or nearly silent (centered RMS {rms:.3g})."
        )
    return signal


def audio_to_tensor(audio_path):
    """Build the CPU float32 tensor with batch and channel dimensions."""
    spectrogram = preprocess_audio(load_audio(audio_path))
    tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
    if tuple(tensor.shape) != EXPECTED_TENSOR_SHAPE:
        raise ValueError(
            f"Unexpected input shape {tuple(tensor.shape)}; "
            f"expected {EXPECTED_TENSOR_SHAPE}. Check preprocessing parameters."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Preprocessing produced non-finite spectrogram values.")
    return tensor
