# til24-asr

Template for FastAPI-based API server. Features:

- Supports both CPU/GPU-accelerated setups automatically.
- Poetry for package management.
- Ruff for formatting & linting.
- VSCode debugging tasks.
- Other QoL packages.

Oh yeah, this template should work with the fancy "Dev Containers: Clone Repository
in Container Volume..." feature.

## Tips from J-H

- Use `apt` for whatever additional dependencies but please record in both `Dockerfile`
  and `.devcontainer/hooks/postCreate.sh` for consistency.
- We might not need `torch` or `torchaudio` if we use Whisper pretrained models
  as there are binary inference runtimes that don't require PyTorch.
- Training should be in the same repository to be neat, but put the training dependencies
  in a separate Poetry dependency group if not needed for inference.

## Input

Audio file provided in `.wav` format with a sample rate of 16 kHz.

Example: <https://github.com/TIL-24/til-24-base/assets/162278270/5e42363d-9f01-4626-8d70-cc7dc8e48c71>

Note that the above example is in `mp4` format as GitHub does not support embedding `.wav` files in README files. However, audio files provided on GCP will be `.wav` files.

## Output

Transcription of audio file. Example: `"Heading is one five zero, target is green commercial aircraft, tool to deploy is electromagnetic pulse."`. For ASR, the server port must be 5001.

## Useful Commands

```sh
# The venv auto-activates, so these work.
poe prod # Launch "production" server.
poe dev # Launch debugging server, use VSCode's debug task instead by pressing F5.
poe test # Run test stolen from the official competition template repo.

# Building docker image for deployment.
docker build -f Dockerfile . -t nyanplan3-asr:latest -t nyanplan3-asr:0.1.0

# Running FastAPI app (with GPU).
docker run --rm --gpus all -p 5001:5001 nyanplan3-asr
```
