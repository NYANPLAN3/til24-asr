"""Singleton for ASR processing."""

import io

import librosa
import numpy as np
import whisper
from faster_whisper import WhisperModel


class ASRManager:
    """ASRManager."""

    def __init__(self):
        """Initialize ASRManager models & stuff."""
        # self.model = whisper.load_model("./models/large-v3.pt")
        self.model = WhisperModel(
            "Systran/faster-distil-whisper-large-v3",
            device="cuda",
            # compute_type="int8_float16",
        )
        self.options = dict(
            language="en",
            compression_ratio_threshold=10.0,
            log_prob_threshold=-10.0,
            no_speech_threshold=1.0,
            beam_size=5,
            patience=1,
            without_timestamps=True,
        )

    async def transcribe(self, wav: bytes) -> str:
        """Transcribe audio bytes to text."""
        # Load the audio bytes to byte stream
        byte_stream = io.BytesIO(wav)

        # Byte stream to audio waveform
        audio_waveform, sr = librosa.load(
            byte_stream, sr=None
        )  # Preserve original sample rate

        # Audio waveform to a numpy array
        audio_waveform = np.array(audio_waveform, dtype=np.float32)

        # Do transcription
        segments, _ = self.model.transcribe(audio_waveform, **self.options)
        segments = list(segments)
        text = segments[0].text

        return text
