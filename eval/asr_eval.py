"""Taken from https://github.com/TIL-24/til-24-base/blob/main/scoring/asr_eval.py."""

from typing import List

import jiwer

wer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " "}),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def asr_eval(truth: List[str], hypothesis: List[str]) -> float:
    result = jiwer.wer(
        truth,
        hypothesis,
        truth_transform=wer_transforms,
        hypothesis_transform=wer_transforms,
    )
    return 1 - result
