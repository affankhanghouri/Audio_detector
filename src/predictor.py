"""Load and validate one CPU model, then reuse it for audio predictions."""

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import torch

from . import config
from .audio_processor import audio_to_tensor
from .model import BearingCNN


class BearingPredictor:
    def __init__(self, model_path=config.MODEL_PATH, cpu_threads=config.CPU_THREADS):
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise ValueError("CPU threads must be a positive integer.")
        torch.set_num_threads(cpu_threads)
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {self.model_path}")
        try:
            checkpoint = torch.load(
                self.model_path, map_location="cpu", weights_only=True,
            )
        except Exception as exc:
            raise ValueError(f"Cannot load checkpoint safely: {exc}") from exc
        self._validate_metadata(checkpoint)
        self.model = BearingCNN()
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Checkpoint does not match BearingCNN: {exc}") from exc
        self.model.eval()
        self.metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}

    @staticmethod
    def _validate_metadata(checkpoint):
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint must contain a model state and training metadata.")
        expected = {
            "sample_rate": config.SAMPLE_RATE,
            "duration": config.DURATION,
            "n_fft": config.N_FFT,
            "hop_length": config.HOP_LENGTH,
            "n_mels": config.N_MELS,
            "top_db": config.TOP_DB,
            "class_mapping": {name: index for index, name in config.CLASS_NAMES.items()},
        }
        for key, value in expected.items():
            if key not in checkpoint or checkpoint[key] != value:
                raise ValueError(
                    f"Checkpoint metadata mismatch for '{key}': "
                    f"expected {value!r}, got {checkpoint.get(key)!r}."
                )
        if not isinstance(checkpoint.get("model_state_dict"), Mapping):
            raise ValueError("Checkpoint is missing a valid model_state_dict.")

    def predict(self, audio_path):
        """Return probabilities and measured seconds; model initialization is excluded."""
        start = perf_counter()
        tensor = audio_to_tensor(audio_path)
        preprocessing_end = perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.shape != (1, 3) or not torch.isfinite(logits).all():
                raise ValueError("Model produced invalid output logits.")
            probabilities = torch.softmax(logits, dim=1)[0]
        inference_end = perf_counter()
        scores = probabilities.tolist()
        predicted_index = int(probabilities.argmax().item())
        result = {
            "audio_path": str(Path(audio_path).expanduser().resolve()),
            "predicted_class": config.CLASS_NAMES[predicted_index],
            "confidence": scores[predicted_index],
            "probabilities": {name: scores[index] for index, name in config.CLASS_NAMES.items()},
            "preprocessing_time": preprocessing_end - start,
            "inference_time": inference_end - preprocessing_end,
        }
        result["total_time"] = perf_counter() - start
        return result
