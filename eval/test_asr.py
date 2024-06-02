"""Taken from https://github.com/TIL-24/til-24-base/blob/main/test_asr.py."""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from eval.asr_eval import asr_eval

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")
HOME = os.getenv("HOME")


def main():
    # input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
    input_dir = Path(f"{HOME}/{TEAM_TRACK}")
    # results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    instances = []

    with open(input_dir / "asr.jsonl", "r") as f:
        for line in tqdm(f.readlines()):
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "audio" / instance["audio"], "rb") as file:
                audio_bytes = file.read()
                instances.append(
                    {**instance,
                        "b64": base64.b64encode(audio_bytes).decode("ascii")}
                )

    results = run_batched(instances)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "asr_results.csv", index=False)
    # calculate eval
    eval_result = asr_eval(
        [result["transcript"] for result in results],
        [result["prediction"] for result in results],
    )
    print(f"1-WER: {eval_result}")


def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 1
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index: index + batch_size]
        response = requests.post(
            "http://localhost:5001/stt",
            # "http://172.17.0.1:5001/stt",
            data=json.dumps(
                {
                    "instances": [
                        {"key": _instance["key"], "b64": _instance["b64"]}
                        for _instance in _instances
                    ]
                }
            ),
        )
        _results = response.json()["predictions"]
        for i in range(len(_instances)):
            pred, gt = _results[i], _instances[i]["transcript"]
            score = asr_eval([gt], [pred])
            if score < 0.995:
                tqdm.write(f'{score:.3f} Pred: {pred}, True: {gt}')
            results.append(
                {
                    "key": _instances[i]["key"],
                    "transcript": gt,
                    "prediction": pred,
                    "score": score,
                }
            )
    return results


if __name__ == "__main__":
    main()
