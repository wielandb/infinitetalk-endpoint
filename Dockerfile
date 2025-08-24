# CUDA Runtime + Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git ffmpeg wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# InfiniteTalk Code ziehen
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git
WORKDIR /workspace/InfiniteTalk
# Repo-Requirements
RUN pip install -r requirements.txt || true

# Zurück ins App-Verzeichnis und unsere App-Dateien kopieren
WORKDIR /workspace
COPY app.py download_weights.py generate.sh ./
RUN chmod +x generate.sh

# Modelle beim Build (optional; spart Kaltstart). Wenn Token nötig, bei Build setzen:
# docker build --build-arg HF_TOKEN=xxxx ...
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN mkdir -p /cache/hf /workspace/weights && python download_weights.py || true

# Expose
EXPOSE 8000

# Start
CMD ["/workspace/generate.sh"]
