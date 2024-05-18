"""Singleton for ASR processing."""

import io

import librosa
import numpy as np

# import whisper
from faster_whisper import WhisperModel


class ASRManager:
    """ASRManager."""

    def __init__(self):
        """Initialize ASRManager models & stuff."""
        # self.model = whisper.load_model("./models/large-v3.pt")
        self.model = WhisperModel(
            "./models/Systran_faster-distil-whisper-large-v3",
            #"./models/whisper-large-v3-atco2-asr-atcosim",
            device="cuda",
            # compute_type="int8_float16",
            compute_type="default",
            local_files_only=True,
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
            
        # Resample the audio to 16000 Hz
        target_sr = 16000
        if sr != target_sr:
            audio_waveform = librosa.resample(audio_waveform, orig_sr=sr, target_sr=target_sr)

        # Convert to mono if needed
        if audio_waveform.ndim > 1:
            audio_waveform = np.mean(audio_waveform, axis=1)

        # Normalize the audio
        audio_waveform = audio_waveform / np.max(np.abs(audio_waveform))
        
        # Audio waveform to a numpy array
        audio_waveform = np.array(audio_waveform, dtype=np.float32)

        # Do transcription
        segments, _ = self.model.transcribe(audio_waveform, **self.options)
        # NOTE: we assume input doesn't exceed model context window
        text = next(segments).text

        return text
