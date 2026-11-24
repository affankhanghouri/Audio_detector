"""Terminal entry point for single-file, recursive-folder, or interactive inference."""

import argparse
from pathlib import Path
import sys

from src.config import CLASS_NAMES, CPU_THREADS, MODEL_PATH, SUPPORTED_AUDIO_SUFFIXES
from src.predictor import BearingPredictor


def display_result(result, actual_class=None):
    print("\n" + "=" * 44)
    print("ACOUSTIC BEARING FAULT DETECTION")
    print("=" * 44)
    print(f"\nAudio File: {result['audio_path']}")
    if actual_class is not None:
        print(f"Actual Condition: {actual_class}")
    print(f"Predicted Condition: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for name, probability in result["probabilities"].items():
        print(f"{name:17} : {probability:7.2%}")
    print(f"\nPreprocessing Time: {result['preprocessing_time'] * 1000:.2f} ms")
    print(f"CNN Inference Time: {result['inference_time'] * 1000:.2f} ms")
    print(f"Total Latency:      {result['total_time'] * 1000:.2f} ms")
    print("=" * 44)


def actual_class_for(path):
    """Recognize the nearest ancestor named after one of the three classes."""
    labels = {name.lower().replace(" ", "_"): name for name in CLASS_NAMES.values()}
    for parent in path.parents:
        label = labels.get(parent.name.lower().replace(" ", "_"))
        if label is not None:
            return label
    return None


def predict_file(predictor, path, actual_class=None):
    try:
        result = predictor.predict(path)
        display_result(result, actual_class)
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error processing {path}: {exc}", file=sys.stderr)
        return None


def predict_folder(predictor, folder):
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: audio directory does not exist: {folder}", file=sys.stderr)
        return 1
    paths = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not paths:
        print(f"No WAV or MP3 files found in {folder}.", file=sys.stderr)
        return 1
    successful = failed = labelled = correct = failed_labelled = 0
    for path in paths:
        actual = actual_class_for(path)
        result = predict_file(predictor, path, actual)
        if result is None:
            failed += 1
            failed_labelled += actual is not None
            continue
        successful += 1
        if actual is not None:
            labelled += 1
            correct += result["predicted_class"] == actual
    print(f"\nFolder summary: {successful} succeeded, {failed} failed, {len(paths)} found.")
    if labelled:
        print(f"Labelled sample accuracy: {correct}/{labelled} ({correct / labelled:.2%}).")
    else:
        print("Sample accuracy unavailable: no successfully processed class-labelled files.")
    print(f"Unlabelled predictions: {successful - labelled}; failed labelled files excluded: {failed_labelled}.")
    return int(failed > 0)


def main():
    parser = argparse.ArgumentParser(description="CPU acoustic bearing fault prediction.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", type=Path, help="WAV or MP3 file to predict")
    source.add_argument("--folder", type=Path, help="Recursively predict WAV and MP3 files")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Trained checkpoint path")
    parser.add_argument("--threads", type=int, default=CPU_THREADS, help="PyTorch CPU threads (default: 2)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        predictor = BearingPredictor(args.model, args.threads)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot initialize predictor: {exc}", file=sys.stderr)
        return 1
    print("Softmax confidence is not a guarantee of diagnostic correctness.")
    if args.audio is not None:
        return int(predict_file(predictor, args.audio) is None)
    if args.folder is not None:
        return predict_folder(predictor, args.folder)

    print("Enter an audio file path, or 'exit' to quit.")
    while True:
        try:
            text = input("Audio path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() == "exit":
            break
        if text:
            # Accept quoted paths copied from a shell, including paths with spaces.
            predict_file(predictor, Path(text.strip("\"'")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPrediction interrupted.", file=sys.stderr)
        raise SystemExit(130)
