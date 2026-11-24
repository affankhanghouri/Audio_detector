# Acoustic Bearing Fault Detection using a 2D CNN

CPU inference for a PyTorch model trained on the SUBF v2.0 acoustic bearing
fault dataset. The three output classes are Normal, Inner Race Fault, and
Outer Race Fault. The application reuses the trained weights; it does not train a model.

The current `Audio_Detector` workspace is the project root (the
`Bearing-Fault-Detection/` directory in the requested layout).

```text
Audio_Detector/
    models/best_bearing_model.pth
    audio_samples/
        Normal/                 # three WAV samples
        Inner_Race_Fault/       # three WAV samples
        Outer_Race_Fault/       # three WAV samples
    src/
        __init__.py
        config.py
        model.py
        audio_processor.py
        predictor.py
    main.py
    app.py                      # simple audio upload interface
    .streamlit/config.toml      # dashboard theme
    tests/test_dashboard.py     # real-model and UI integration checks
    requirements.txt
    README.md
    .gitignore
```

The original root-level checkpoint and `bearing_audio_samples ` directory
(whose name ends with a space) are preserved. The organized copies are used by
the application. No weights or audio were generated or replaced.

## Installation

Target: Ubuntu 24.04, Python 3.12, CPU inference, two PyTorch threads by default.
The existing `.venv` contains CPU PyTorch and should be reused:

```bash
cd /home/affan/Desktop/Audio_Detector
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a fresh environment, install CPU PyTorch explicitly before other dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

No CUDA packages are needed. SoundFile wheels normally supply libsndfile. If
your system install requires it, install `libsndfile1` using Ubuntu's package
manager. MP3 support depends on the installed decoder; an FFmpeg installation
can support Librosa's decoder fallback. WAV decoding is preferred for reproducing
the supplied evaluation input.

## Execution

```bash
python main.py --audio audio_samples/Normal/sample_1.wav
python main.py --folder audio_samples
python main.py
python main.py --folder audio_samples --threads 2
python main.py --audio /absolute/path/recording.mp3 --model models/best_bearing_model.pth
```

With no input flag, enter audio paths repeatedly and type `exit` to finish.
The model loads once per process and is reused. `--audio` and `--folder` are
mutually exclusive. Directory mode finds WAV/MP3 files recursively, accepts
uppercase suffixes, continues after individual errors, and returns a nonzero
exit code if any file fails. Normal, Inner_Race_Fault, and Outer_Race_Fault
ancestor folder names supply ground-truth labels (spaces are also accepted).
Accuracy uses successfully predicted labelled files; failed and unlabelled
files are counted separately.

The default checkpoint path is absolute and derived from `src/config.py`, so
it works from any current directory. User-supplied relative audio, folder, and
model paths are resolved from the shell's current directory, as usual:

```bash
/home/affan/Desktop/Audio_Detector/.venv/bin/python \
  /home/affan/Desktop/Audio_Detector/main.py \
  --audio /home/affan/Desktop/Audio_Detector/audio_samples/Normal/sample_1.wav
```

## Simple Streamlit app

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the Local URL printed by Streamlit, normally `http://localhost:8501`.
Upload a WAV or MP3 recording and click **Analyze audio**. The app displays
one of three predicted classes: **Normal**, **Inner Race Fault**, or
**Outer Race Fault**, with its confidence. Expand **View all class probabilities**
to see the scores for all three classes. An audio player lets you listen to the
uploaded recording.

Recordings must be at least ten seconds long; only the first ten seconds are
analyzed. Invalid, silent, corrupt, or short recordings produce a readable error.
The model loads on the first analysis and is cached across uploads and reruns.
Results persist for the current upload; changing or removing the upload clears
the previous result. Temporary files are cleaned up after analysis, including
when prediction fails.

The app uses the existing `BearingPredictor` and its unchanged preprocessing
and weights. The CLI remains available. The interface has no sidebar, plots,
sample selectors, or prediction history, and no longer requires Matplotlib.
The model remains an academic prototype trained on SUBF data and has not been
validated on independently recorded industrial machinery. Softmax confidence
does not guarantee diagnostic correctness.

Run integration checks with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These checks cover real-model prediction parity for all nine WAV recordings,
upload interactions, result persistence, model caching, invalid audio,
missing checkpoints, temporary-file cleanup, and the existing CLI.

## Audio and preprocessing

Inputs must be decodable WAV or MP3 recordings of at least ten seconds.
Librosa loads mono float32 audio at its native rate, verifies its duration, then
resamples to 10,000 Hz; only the first ten seconds (100,000 samples) are used.
Checking duration before resampling prevents rounding a short input upward.
Short inputs are rejected without padding.
Empty, non-finite, silent, and nearly silent inputs are rejected. The silence
threshold is centered RMS <= 1e-6 in decoded floating-point amplitude. This check
also rejects constant DC signals and does not modify accepted audio.

Training preprocessing is reproduced in `preprocess_audio`:

1. Subtract the signal mean.
2. Compute a power mel spectrogram: FFT 1024, hop 256, 128 mel bands,
   fmin 0, fmax 5000, power 2.0. Librosa defaults include centered frames,
   a Hann window, constant padding, and Slaney mel normalization.
3. Convert power to dB with `ref=np.max`, `amin=1e-20`, and `top_db=80`.
4. Clip `(log_mel + 80) / 80` to [0, 1] and cast to float32.
5. Add batch and channel axes to obtain `(1, 1, 128, 391)`.

There is no image resizing, RGB conversion, PNG encoding, or additional
normalization. Resampling/downmixing occur before this training function.
The original training library versions were not supplied, so exact numerical
equivalence across different Librosa versions cannot be established from the
checkpoint alone.

## Model and prediction

`src/model.py` preserves the saved architecture and module indices:

```text
1 input channel
Conv2d(1,16,3,padding=1)   -> BatchNorm -> ReLU -> MaxPool(2)
Conv2d(16,32,3,padding=1)  -> BatchNorm -> ReLU -> MaxPool(2)
Conv2d(32,64,3,padding=1)  -> BatchNorm -> ReLU -> MaxPool(2)
Conv2d(64,128,3,padding=1) -> BatchNorm -> ReLU -> MaxPool(2)
AdaptiveAvgPool(4,4) -> Flatten -> Linear(2048,128)
ReLU -> Dropout(0.3) -> Linear(128,3)
```

`BearingPredictor` safely loads the dictionary checkpoint with
`map_location="cpu", weights_only=True`, verifies all six preprocessing
metadata fields and the exact class mapping, strictly loads the state dictionary,
and calls `eval()`. Predictions use `torch.inference_mode()` and softmax.
Thread count is configurable through `--threads` or the constructor; PyTorch's
thread setting applies to the process.

```python
from src.predictor import BearingPredictor

predictor = BearingPredictor(cpu_threads=2)
result = predictor.predict("audio_samples/Normal/sample_1.wav")
print(result)
```

Results contain the absolute `audio_path`, `predicted_class`, `confidence`,
all three `probabilities`, and `preprocessing_time`, `inference_time`, and
`total_time` in seconds. Terminal times are in milliseconds. Timings use
`time.perf_counter()`. Preprocessing includes loading, validation and feature
extraction; inference includes the CNN, output validation and softmax; total
also includes result construction. Model initialization is excluded. The first
prediction may be substantially slower due to Librosa imports and compilation.

## Existing Kaggle evaluation

These are the supplied training/evaluation results, rather than a new local
benchmark:

| Metric | Result |
| --- | ---: |
| Training accuracy | 94.42% |
| Validation accuracy | 97.96% |
| Test accuracy | 89.44% |
| Training epochs | 1 |

The checkpoint records epoch 1 and validation accuracy 0.9796296296296296,
consistent with the reported rounded validation result.

## Limitations

The trained model was evaluated on SUBF data and has not yet been validated on
independently recorded industrial machinery. Recording hardware, distance,
background noise, speed, load, and compression can change predictions. The
classifier always chooses one of its three classes; it has no unknown-fault
class. Softmax confidence is not calibrated diagnostic certainty and does not
guarantee correctness. Local accuracy on the nine supplied samples is a small
sample check and does not replace the held-out Kaggle test evaluation.

Only the first ten seconds affect a prediction. MP3 codec delay and loss may
affect features. Measured CPU latency depends on hardware, background load,
library initialization, and thread settings.

## Local verification (25 November 2026)

Verified on the Intel Core i5-6300U with two PyTorch CPU threads. The real
checkpoint loaded successfully and all nine original WAV samples ran without
errors. Overall folder-label accuracy was **7/9 (77.78%)**. The two errors
were Inner_Race_Fault/sample_2.wav predicted as Normal, and
Normal/sample_2.wav predicted as Inner Race Fault.

| Audio sample | Predicted class | Confidence | Preprocessing ms | CNN + softmax ms | Total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Inner_Race_Fault/sample_1.wav | Inner Race Fault | 99.89% | 4560.08 | 126.48 | 4686.85 |
| Inner_Race_Fault/sample_2.wav | Normal | 77.23% | 78.00 | 194.02 | 272.26 |
| Inner_Race_Fault/sample_3.wav | Inner Race Fault | 99.55% | 39.86 | 80.86 | 120.96 |
| Normal/sample_1.wav | Normal | 99.98% | 86.55 | 97.84 | 186.09 |
| Normal/sample_2.wav | Inner Race Fault | 92.88% | 77.38 | 109.71 | 187.45 |
| Normal/sample_3.wav | Normal | 99.87% | 72.69 | 134.96 | 207.88 |
| Outer_Race_Fault/sample_1.wav | Outer Race Fault | 99.95% | 100.33 | 144.07 | 244.74 |
| Outer_Race_Fault/sample_2.wav | Outer Race Fault | 99.81% | 82.44 | 56.51 | 139.23 |
| Outer_Race_Fault/sample_3.wav | Outer Race Fault | 99.91% | 23.59 | 98.73 | 127.59 |

These are observed single-run timings, including the first call in this process;
they are not statistical latency estimates. The first-ever prediction immediately
after installing Librosa took about 26.56 seconds because of initialization and
compilation. Later runs benefit from cached code. Background load also affects timings.

All **27 verification checks** passed. They covered imports, safe checkpoint
loading and metadata validation, CPU/eval mode, tensor/output shapes, byte-for-byte
preservation of the originals, nine-sample inference, short/silent/non-finite/corrupt
input rejection, a file one native frame shorter than ten seconds, stereo
22.05 kHz resampling and cropping, folder error continuation, missing model/audio
handling, metadata/class-mapping errors, thread validation, MP3 inference, and
single-file and repeated interactive execution from outside the project directory.
The full recursive CLI separately reported nine successes, zero failures and 7/9
labelled accuracy. `python -m pip check` reported no broken requirements.

Verified library versions: torch 2.15.0+cpu, numpy 2.5.4, librosa 0.11.0, soundfile 0.13.1.

Full probabilities, per-file timings in seconds, and individual check descriptions
are saved in [verification_results.json](verification_results.json). The audio
validation fixtures used for verification were temporary and are not part of the
sample dataset. No model retraining or artificial checkpoints were used.
