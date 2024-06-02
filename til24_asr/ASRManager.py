"""Singleton for ASR processing."""

from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import List

import torch
from dotenv import load_dotenv

# Force import in specific order.
if True:
    load_dotenv()
    from nemo.collections.asr.models.aed_multitask_models import (
        EncDecMultiTaskModel,
        MultiTaskDecodingConfig,
    )
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

DEVICE = "cuda"


# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
class ASRManager:
    """ASRManager."""

    def __init__(self):
        """Initialize ASRManager models & stuff."""
        self._init_parakeet()
        self.model.eval().freeze()

    def _init_canary(self):
        assert False
        model = EncDecMultiTaskModel.from_pretrained(
            'nvidia/canary-1b', map_location=DEVICE)
        sampling: MultiTaskDecodingConfig = model.cfg.decoding
        sampling.beam.beam_size = 1
        model.change_decoding_strategy(sampling)
        self.model: EncDecMultiTaskModel = model
        self.is_rnnt = False

    def _init_parakeet(self):
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
        self.model: EncDecRNNTBPEModel = model
        self.is_rnnt = True

    @torch.inference_mode()
    @torch.autocast(DEVICE)
    async def transcribe(self, wavs: List[bytes]) -> List[str]:
        """Transcribe audio bytes to text."""
        with ExitStack() as stack:
            files = [stack.enter_context(
                NamedTemporaryFile(suffix=".wav")) for _ in wavs]
            for f, w in zip(files, wavs):
                f.write(w)
            if self.is_rnnt:
                texts, _ = self.model.transcribe(
                    [f.name for f in files], batch_size=len(wavs))
            else:
                texts = self.model.transcribe(
                    [f.name for f in files], batch_size=1)
        return texts
