# syntax=docker/dockerfile:1

ARG CUDA_VERSION=12.1.1

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 as deploy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
  echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends \
  libcudnn8 \
  libcublas-12-1 \
  curl
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends python3-pip
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -U pip

WORKDIR /app

# Remember to regenerate requirements.txt!
COPY --link requirements.txt .env ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -r requirements.txt

COPY --link models ./models
COPY --link til24_asr ./til24_asr

EXPOSE 5001
# uvicorn --host=0.0.0.0 --port=5001 --factory til24_asr:create_app
CMD ["uvicorn", "--host=0.0.0.0", "--port=5001", "--factory", "til24_asr:create_app"]
