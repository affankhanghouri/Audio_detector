"""Training parameters and paths anchored to this project's location."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_bearing_model.pth"
AUDIO_SAMPLES_PATH = PROJECT_ROOT / "audio_samples"

SAMPLE_RATE = 10000
DURATION = 10
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
TOP_DB = 80
CPU_THREADS = 2
EXPECTED_TENSOR_SHAPE = (1, 1, 128, 391)
# Only used to reject uninformative input; it does not normalize the signal.
MIN_CENTERED_RMS = 1e-6

CLASS_NAMES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
}

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3"}
