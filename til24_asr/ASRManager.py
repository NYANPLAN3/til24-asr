"""Singleton for ASR processing."""

from tempfile import NamedTemporaryFile

import torch
from dotenv import load_dotenv

# Force import in specific order.
if True:
    load_dotenv()
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

DEVICE = "cuda"


class ASRManager:
    """ASRManager."""

    def __init__(self):
        """Initialize ASRManager models & stuff."""
        model = EncDecRNNTBPEModel.restore_from(
            "./models/parakeet-tdt-1.1b.nemo",
            map_location=DEVICE,
        )
        # NOTE: ValueError: currently only greedy is supported...
        # sampling: RNNTBPEDecodingConfig = model.cfg.decoding
        # sampling.strategy = "beam"
        # sampling.beam.beam_size = 2
        # sampling.beam.return_best_hypothesis = True
        # model.change_decoding_strategy(sampling)
        self.model: EncDecRNNTBPEModel = model.eval()
        self.model.freeze()

    @torch.inference_mode()
    @torch.autocast(DEVICE)
    async def transcribe(self, wav: bytes) -> str:
        """Transcribe audio bytes to text."""
        with NamedTemporaryFile(suffix='.wav') as f:
            f.write(wav)
            f.flush()
            texts, _ = self.model.transcribe([f.name])
            text = texts[0]

        return text
