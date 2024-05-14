"""Singleton for ASR processing."""

import whisper
import io
import numpy as np
import librosa

class ASRManager:
    """ASRManager."""

    def __init__(self):
        """Initialize ASRManager models & stuff."""
        self.model = whisper.load_model("large-v3")

    async def transcribe(self, wav: bytes) -> str:
        """Transcribe audio bytes to text."""
        # Load the audio bytes to byte stream
        byte_stream = io.BytesIO(audio_bytes)

        # Byte stream to audio waveform
        audio_waveform, sr = librosa.load(byte_stream, sr=None)  # Preserve original sample rate
        
        # Audio waveform to a numpy array
        audio_waveform = np.array(audio_waveform, dtype=np.float32)
        
        # Do transcription
        result = self.model.transcribe(audio_waveform)
        
        return result['text']
