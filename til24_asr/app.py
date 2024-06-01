"""Main app."""

import base64
import logging

# import enchant
import os
import re
import sys

# from enchant.checker import SpellChecker
from dotenv import load_dotenv
from fastapi import FastAPI
from num2words import num2words

from .ASRManager import ASRManager
from .log import setup_logging
from .structs import STTRequest

__all__ = ["create_app"]

load_dotenv()

setup_logging
log = logging.getLogger(__name__)


def create_app():
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
            debug["cuda_device"] = str(
                torch.zeros([10, 10], device="cuda").device)
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

    def capitalize_start_of_sentence(text):
        def capitalize_match(match):
            return match.group(1) + match.group(2).upper()

        return re.sub(r"(^|[.!?]\s+)([a-z])", capitalize_match, text)

    # fmt: off
    def process_output(o):
        # Add a space between a digit and a letter / letter and a digit
        o = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", o)
        o = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", o)
        o = re.sub(r"(?<=\d)(?=\d)", " ", o) # space between digits

        # Add period at the end of text
        o = re.sub(r"([^.,])([.,])$", r"\1.", o)
        o = re.sub(r"(?<=\w)\.(?=\w)", " ", o)

        # Correct some number stuff before converting to words
        # o = re.sub(r" +", " ", o)
        # o = re.sub(r"\bheading 2 (\d \d \d)\b", r"heading to \1", o, flags=re.IGNORECASE)

        # Numbers to words
        o = re.sub(r"(\d+)", lambda m: num2words(int(m.group())), o)

        o = re.sub(r"\b(ground)\b", r"brown", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(machine guns)\b", r"machine gun", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(fighter jets)\b", r"fighter jet", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(capture)\b", r"catcher", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(nine)\b", r"niner", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(torret)\b", r"turret", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(torrid)\b", r"turret", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(turrell)\b", r"turret", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(engate)\b", r"engage", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(gray)\b", r"grey", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(intercepted)\b", r"interceptor", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(engaged)\b", r"engage", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(hostel)\b", r"hostile", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(Heading)\b", r"heading", o)
        o = re.sub(r"\b(anterior)\b", r"anti-air", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(anti air)\b", r"anti-air", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(surface to air)\b", r"surface-to-air", o, flags=re.IGNORECASE)
        # idk
        o = re.sub(r"\b(great)\b", r"red", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(grate)\b", r"red", o, flags=re.IGNORECASE)
        o = re.sub(r"\b(rate)\b", r"red", o, flags=re.IGNORECASE)

        # US English to UK English
        # output = us_spelling_to_uk(output)

        # Capitalize first letter
        o = capitalize_start_of_sentence(o)

        # Remove extra spaces
        o = re.sub(r" +", " ", o)
        o = o.strip()
        o = o[0].upper() + o[1:]

        return o

    # fmt: on
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

    return app
