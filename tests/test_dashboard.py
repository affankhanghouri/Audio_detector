"""Integration checks for the simple upload interface and the real model."""

import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
from src.config import AUDIO_SAMPLES_PATH, PROJECT_ROOT
from src.predictor import BearingPredictor


def wav_bytes(signal):
    output = io.BytesIO()
    sf.write(output, signal, 10000, format="WAV", subtype="FLOAT")
    return output.getvalue()


class DashboardTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def make_app(self):
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90).run()
        self.assertFalse(at.exception)
        return at

    def upload_sample(self, at, name="sample_1.wav"):
        path = AUDIO_SAMPLES_PATH / "Normal" / name
        at.file_uploader[0].set_value((name, path.read_bytes(), "audio/wav")).run()

    def test_all_nine_samples_match_cli(self):
        predictor = BearingPredictor()
        paths = sorted(AUDIO_SAMPLES_PATH.rglob("*.wav"))
        self.assertEqual(len(paths), 9)
        for path in paths:
            with self.subTest(sample=str(path)):
                uploaded = app.analyze_recording(path.name, path.read_bytes())
                cli = predictor.predict(path)
                self.assertEqual(uploaded["predicted_class"], cli["predicted_class"])
                np.testing.assert_allclose(list(uploaded["probabilities"].values()), list(cli["probabilities"].values()), atol=1e-7)

    def test_upload_results_and_cached_model(self):
        original_init, original_predict = BearingPredictor.__init__, BearingPredictor.predict
        with patch.object(BearingPredictor, "__init__", autospec=True, side_effect=original_init) as initialized, patch.object(BearingPredictor, "predict", autospec=True, side_effect=original_predict) as inferred:
            at = self.make_app()
            self.assertTrue(at.button[0].disabled)
            self.assertEqual(len(at.sidebar), 0)
            self.assertEqual(initialized.call_count, 0)
            self.upload_sample(at)
            self.assertEqual(len(at.get("audio")), 1)
            self.assertEqual(inferred.call_count, 0)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["prediction"]["predicted_class"], "Normal")
            self.assertEqual(len(at.get("progress")), 3)
            result = at.session_state["prediction"]
            at.run()
            self.assertEqual(at.session_state["prediction"], result)
            self.assertEqual(initialized.call_count, 1)
            self.assertEqual(inferred.call_count, 1)
            self.upload_sample(at, "sample_2.wav")
            self.assertIsNone(at.session_state["prediction"])
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(initialized.call_count, 1)
            self.assertEqual(inferred.call_count, 2)
            at.file_uploader[0].clear().run()
            self.assertIsNone(at.session_state["prediction"])
            self.assertTrue(at.button[0].disabled)

    def test_invalid_audio_and_missing_checkpoint(self):
        at = self.make_app()
        for name, data, message in (
            ("silence.wav", wav_bytes(np.zeros(100000, dtype=np.float32)), "silent"),
            ("short.wav", wav_bytes(np.ones(99999, dtype=np.float32)), "at least 10 seconds"),
            ("broken.wav", b"not audio", "Cannot decode"),
        ):
            with self.subTest(upload=name):
                at.file_uploader[0].set_value((name, data, "audio/wav")).run()
                at.button[0].click().run()
                self.assertFalse(at.exception)
                self.assertIsNone(at.session_state["prediction"])
                self.assertTrue(any(message in error.value for error in at.error))
        st.cache_resource.clear()
        self.upload_sample(at)
        with patch.object(BearingPredictor, "__init__", side_effect=FileNotFoundError("Trained checkpoint not found")):
            at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("checkpoint not found" in error.value for error in at.error))

    def test_temporary_upload_cleanup(self):
        with app.audio_file("../../upload.wav", b"test") as path:
            self.assertEqual(path.name, "recording.wav")
            directory = path.parent
        self.assertFalse(directory.exists())
        with self.assertRaises(RuntimeError):
            with app.audio_file("upload.wav", b"test") as path:
                directory = path.parent
                raise RuntimeError("Test failure")
        self.assertFalse(directory.exists())
        with self.assertRaises(ValueError):
            with app.audio_file("upload.txt", b"test"):
                self.fail("Unsupported format accepted")

    def test_cli_still_runs(self):
        path = AUDIO_SAMPLES_PATH / "Normal" / "sample_1.wav"
        process = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "--audio", str(path)], cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Predicted Condition: Normal", process.stdout)


if __name__ == "__main__":
    unittest.main()
