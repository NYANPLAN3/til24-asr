"""Main app."""
import base64
import logging
#import enchant
import os
import re
import sys

from num2words import num2words
#from enchant.checker import SpellChecker

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

"""def us_spelling_to_uk(text, checker = enchant.checker.SpellChecker("en_GB"), ignore_list = []):
    checker.set_text(text)
    for err in checker:
        suggestions = err.suggest()
        if err.word in ignore_list:
            continue
        if len(suggestions) < 1:
            #print(err.word)
            continue
        # else: print(name, "W:", err.word, "\tC:", suggestions[0])
        err.replace(suggestions[0])
    return checker.get_text()
"""
    
def process_output(output):

    # Add a space between a digit and a letter / letter and a digit
    output = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', output)
    output = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', output)
    output = re.sub(r'(?<=\d)(?=\d)', ' ', output)

    # Add period at the end of text
    output = re.sub(r'([^.,])([.,])$', r'\1.', output)
    
    # Numbers to words
    output = re.sub(r'(\d+)', lambda m: num2words(int(m.group())), output)

    output = re.sub(r'\b(nine)\b', r'niner', output, flags=re.IGNORECASE)
    output = re.sub(r'\b(torret)\b', r'turret', output, flags=re.IGNORECASE)
    output = re.sub(r'\b(engate)\b', r'engage', output, flags=re.IGNORECASE)
    # US English to UK English
    #output = us_spelling_to_uk(output)

    # Remove extra spaces 
    output = re.sub(r' +', ' ', output)
    output = output.strip()
    
    # Capitalize first letter
    output = output[0].upper() + output[1:]

    return output

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
        text = process_output(text)
        preds.append(text)

    return {"predictions": preds}
