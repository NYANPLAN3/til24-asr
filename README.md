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

The venv auto-activates, so these work.

```sh
# Launch debugging server, use VSCode's debug task instead by pressing F5.
poe dev
# Run test stolen from the official competition template repo.
poe test
# Building docker image for deployment, will also be tagged as latest.
poe build {insert_version_like_0.1.0}
# Run the latest image locally.
poe prod
# Publish the latest image to GCP artifact registry.
poe publish
```

Finally, to submit the image (must be done on GCP unfortunately).

```sh
gcloud ai models upload --region asia-southeast1 --display-name 'nyanplan3-asr' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-nyanplan3/nyanplan3-asr:finals --container-health-route /health --container-predict-route /stt --container-ports 5001 --version-aliases default
```

```sh
ct2-transformers-converter --model /workspaces/til24-main/til24-asr/models/experimental-af --output_dir /workspaces/til24-main/til24-asr/models/experimental-af-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

Baseline: 0.9902903959674662
Baseline no prompt: 0.9815770300209656
Ft: 0.9900524883648313
Ft no prompt: 0.9880154045172707
