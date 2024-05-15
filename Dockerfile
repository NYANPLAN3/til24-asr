# syntax=docker/dockerfile:1
# NOTE: Docker's GPU support works for most images despite common misconceptions.
#FROM python:3.11-slim-bookworm
# Example of prebuilt pytorch image to save download time.
#FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Cache packages to speed up builds, see: https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#run---mounttypecache
# Example:
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
  echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends python3-pip

# COPY should be from least changed to most frequently changed.
COPY --link models ./models

COPY --link poetry.lock pyproject.toml README.md .
RUN mkdir til24_asr && touch til24_asr/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -U pip \
  && pip install -e .
  
COPY --link til24_asr ./til24_asr

EXPOSE 5001
CMD ["fastapi", "run", "til24_asr", "--proxy-headers", "--port", "5001"]
