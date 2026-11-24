"""Simple upload-and-predict interface for the existing bearing classifier."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile

import streamlit as st

from src.config import CPU_THREADS, MODEL_PATH, SUPPORTED_AUDIO_SUFFIXES
from src.predictor import BearingPredictor


@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    """Reuse the same CPU model across uploads and page interactions."""
    return BearingPredictor(model_path=MODEL_PATH, cpu_threads=CPU_THREADS)


@contextmanager
def audio_file(filename, audio_bytes):
    """Keep uploads in a temporary directory and remove them after prediction."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError("Please upload a WAV or MP3 file.")
    with tempfile.TemporaryDirectory(prefix="bearing_upload_") as directory:
        path = Path(directory) / f"recording{suffix}"
        path.write_bytes(audio_bytes)
        yield path


def analyze_recording(filename, audio_bytes):
    """Use the existing predictor without changing its preprocessing or weights."""
    with audio_file(filename, audio_bytes) as path:
        result = get_predictor().predict(path)
    result["audio_path"] = filename
    return result


def main():
    st.set_page_config(page_title="Bearing Fault Detection", layout="centered")
    st.title("Acoustic Bearing Fault Detection")
    st.write("Upload an audio recording to predict the bearing condition.")
    st.caption("Normal · Inner Race Fault · Outer Race Fault")

    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3"])
    st.caption("Use a recording of at least 10 seconds. Only the first 10 seconds are analyzed.")

    st.session_state.setdefault("active_recording", None)
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("analysis_error", None)
    audio_bytes = uploaded.getvalue() if uploaded is not None else None
    identity = (
        hashlib.sha256(uploaded.name.encode() + b"\0" + audio_bytes).hexdigest()
        if uploaded is not None else None
    )
    if identity != st.session_state.active_recording:
        # A new upload must never inherit the previous recording's prediction.
        st.session_state.active_recording = identity
        st.session_state.prediction = None
        st.session_state.analysis_error = None

    if uploaded is not None:
        mime = "audio/mpeg" if Path(uploaded.name).suffix.lower() == ".mp3" else "audio/wav"
        st.audio(audio_bytes, format=mime)

    if st.button("Analyze audio", type="primary", disabled=uploaded is None, width="stretch"):
        try:
            with st.spinner("Analyzing audio…"):
                st.session_state.prediction = analyze_recording(uploaded.name, audio_bytes)
            st.session_state.analysis_error = None
        except Exception as exc:
            st.session_state.prediction = None
            st.session_state.analysis_error = str(exc) or "Could not analyze this recording. Please try a valid WAV or MP3."

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)
    result = st.session_state.prediction
    if result is not None:
        st.divider()
        st.caption("PREDICTED CONDITION")
        st.subheader(result["predicted_class"])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        with st.expander("View all class probabilities"):
            for name, probability in result["probabilities"].items():
                st.progress(probability, text=f"{name}: {probability:.2%}")

    st.caption(
        "Academic prototype trained on SUBF data; not validated on independently recorded industrial machinery. "
        "Confidence does not guarantee diagnostic correctness."
    )


if __name__ == "__main__":
    main()
