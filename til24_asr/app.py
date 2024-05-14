"""Main app."""

import base64
import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI

from .ASRManager import ASRManager
from .structs import STTRequest

__all__ = ["app"]

load_dotenv()

log = logging.getLogger(__name__)

app = FastAPI()

asr_manager = ASRManager()


@app.get("/hello")
async def hello():
    """J-H: I added this to dump useful info for debugging.

    Returns:
        dict: JSON message.
    """
    debug = {}
    debug["py_version"] = sys.version
    debug["task"] = "ASR"
    debug["env"] = dict(os.environ)

    try:
        import torch  # type: ignore

        debug["torch_version"] = torch.__version__
        debug["cuda_device"] = str(torch.zeros([10, 10], device="cuda").device)
    except ImportError:
        pass

    return debug


@app.get("/health")
async def health():
    """Competition admin needs this."""
    return {"message": "health ok"}


@app.post("/stt")
async def stt(req: STTRequest):
    """Performs ASR given the filepath of an audio file."""
    # get base64 encoded string of audio, convert back into bytes.

    preds = []
    for instance in req.instances:
        # each is a dict with one key "b64" and the value as a b64 encoded string.
        wav = base64.b64decode(instance.b64)

        # TODO: Consider padding & stacking audio files for batch processing.
        text = await asr_manager.transcribe(wav)
        preds.append(text)

    return {"predictions": preds}
