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
    def process_output(output):
        # Add a space between a digit and a letter / letter and a digit
        output = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", output)
        output = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", output)
        output = re.sub(r"(?<=\d)(?=\d)", " ", output)

        # Add period at the end of text
        output = re.sub(r"([^.,])([.,])$", r"\1.", output)
        output = re.sub(r"(?<=\w)\.(?=\w)", " ", output)

        # Numbers to words
        output = re.sub(r"(\d+)", lambda m: num2words(int(m.group())), output)

        output = re.sub(r"\b(ground)\b", r"brown", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(machine guns)\b", r"machine gun", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(fighter jets)\b", r"fighter jet", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(capture)\b", r"catcher", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(nine)\b", r"niner", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(torret)\b", r"turret", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(torrid)\b", r"turret", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(turrell)\b", r"turret", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(engate)\b", r"engage", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(gray)\b", r"grey", output, flags=re.IGNORECASE)
        output = re.sub(
            r"\b(intercepted)\b", r"interceptor", output, flags=re.IGNORECASE
        )
        output = re.sub(r"\b(engaged)\b", r"engage", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(hostel)\b", r"hostile", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(Heading)\b", r"heading", output)
        output = re.sub(r"\b(anterior)\b", r"anti-air", output, flags=re.IGNORECASE)
        output = re.sub(r"\b(anti air)\b", r"anti-air", output, flags=re.IGNORECASE)
        output = re.sub(
            r"\b(surface to air)\b", r"surface-to-air", output, flags=re.IGNORECASE
        )
        output = re.sub(r"\b(e m p)\b", r"EMP", output, flags=re.IGNORECASE)

        # US English to UK English
        # output = us_spelling_to_uk(output)

        # Capitalize first letter
        output = capitalize_start_of_sentence(output)

        # Remove extra spaces
        output = re.sub(r" +", " ", output)
        output = output.strip()
        output = output[0].upper() + output[1:]

        return output
    # fmt: on

    @app.post("/stt")
    async def stt(req: STTRequest):
        """Performs ASR given the filepath of an audio file."""
        wavs = [base64.b64decode(instance.b64) for instance in req.instances]
        texts = await asr_manager.transcribe(wavs)
        preds = [process_output(text) for text in texts]
        return {"predictions": preds}

    return app
