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
        self.model = WhisperModel(
            "./models/whisper-large-v3-ct2",
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
            initial_prompt=(
                "Engage interceptor jets to intercept an orange commercial aircraft heading three one five. "
                "Control tower to turrets, deploy EMP on white fighter jet heading one niner five. "
                "Alfa, Echo, Mike Papa, deploy yellow drone with surface-to-air missiles. Alpha, deploy surface-to-air missiles at heading two five five. "
                "Alpha, Bravo, Charlie, this is Control Tower. Deploy electromagnetic pulse at heading two six zero towards the black, purple, and orange drone. Target locked. Execute. "
                "Turret Alpha, engage green and orange commercial aircraft at heading zero niner zero with anti-air artillery. Turret Bravo, standby for further instructions. "
            ),
        )

    async def transcribe(self, wav: bytes) -> str:
        """Transcribe audio bytes to text."""
        # Load the audio bytes to byte stream
        byte_stream = io.BytesIO(wav)

        # Byte stream to audio waveform
        # NOTE: This is better than what faster-whisper does if you pass an audio file directly
        wav, _ = librosa.load(
            byte_stream, sr=16000, mono=True, res_type="soxr_vhq"
        )

        # Normalize the audio
        # https://github.com/huggingface/transformers/blob/6bd511a45a58eb02bd59cf447141a2af428747a4/src/transformers/models/whisper/feature_extraction_whisper.py#L176
        wav = (wav - np.mean(wav)) / np.sqrt(np.var(wav) + 1e-7)
        # wav = wav / np.max(np.abs(wav))

        # Do transcription
        segments, _ = self.model.transcribe(wav, **self.options)
        # NOTE: we assume input doesn't exceed model context window
        text = next(segments).text

        return text.strip()
