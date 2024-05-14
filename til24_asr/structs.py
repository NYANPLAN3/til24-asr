"""Types."""

from typing import List

from pydantic import BaseModel

__all__ = ["STTEntry", "STTRequest"]


class STTEntry(BaseModel):
    """Speech-to-text entry."""

    b64: str


class STTRequest(BaseModel):
    """Speech-to-text request."""

    instances: List[STTEntry]
