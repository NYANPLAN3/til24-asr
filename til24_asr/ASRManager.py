"""Singleton for ASR processing."""

import io
import logging

import librosa
import numpy as np

# import whisper
from faster_whisper import WhisperModel

MODEL_PATH = "./models/whisperv2-exp2-ct2"
# MODEL_PATH = "./models/whisper-large-v3-ct2"
# MODEL_PATH = "large-v2"

log = logging.getLogger(__name__)


class ASRManager:
    """ASRManager."""

    def __init__(self):
        """Initialize ASRManager models & stuff."""
        log.info(f"Loading: {MODEL_PATH}")
        self.model = WhisperModel(
            MODEL_PATH,
            device="cuda",
            # compute_type="int8_float16",
            compute_type="default",
            local_files_only=True,
        )
        log.info("ASR loaded.")
        self.options = dict(
            language="en",
            # compression_ratio_threshold=10.0,
            # log_prob_threshold=-10.0,
            # no_speech_threshold=0.6,
            beam_size=5,
            patience=1,
            without_timestamps=True,
            initial_prompt=(
                "Air defense turret, adjust heading to two one five. Deploy surface-to-air missiles to intercept the silver, brown, and grey cargo aircraft."
                "Deploy anti-air artillery, heading one two zero, engage black light aircraft."
                "Set heading to zero seven five, target the red and yellow commercial aircraft, and deploy electromagnetic pulse."
                "Engage interceptor jets to intercept an orange commercial aircraft heading three one five. "
                "Control tower to turrets, deploy EMP tool on white fighter jet heading one niner five. "
                "Deploy anti-air artillery to heading zero niner five. Engage grey, purple, and white light aircraft."
                "Alpha, Bravo, Charlie, this is Control Tower. Deploy electromagnetic pulse at heading one six zero towards the black, purple, and orange drone. Target locked. Execute. "
                "Turret Alpha, engage green and orange commercial aircraft at heading two niner zero with anti-air artillery. Turret Bravo, standby for further instructions. "
                "Control tower to air defense turrets, this is Alpha. Set heading to two niner zero. Target the orange, purple, and black cargo aircraft. Deploy interceptor jets. Repeat, deploy interceptor jets. Over."
            ),
        )

        log.info("Warmup...")
        self._warmup()
        log.info("Hot!")

    def _warmup(self):
        # Need a random wav that the model thinks is text...
        np.random.seed(42)
        wav = np.random.randn(320000)
        wav = (wav - np.mean(wav)) / np.sqrt(np.var(wav) + 1e-7)
        segments, _ = self.model.transcribe(wav, **self.options)
        try:
            text = next(segments).text
            log.debug(f"Hallucination: {text}")
        except StopIteration:
            log.debug("Hallucination failed.")

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
        try:
            text = next(segments).text
        except StopIteration:
            log.error("No text found?")
            return ""

        return text.strip()
