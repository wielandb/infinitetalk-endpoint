#!/usr/bin/env python3
"""
download_weights.py
Lädt alle benötigten Gewichte für InfiniteTalk konsistent unter /workspace/weights.

ENV:
- HF_TOKEN (optional): Token für gated Models
- WEIGHTS_DIR (optional, default: /workspace/weights)
- HF_HOME / HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE (optional, default: /workspace/cache/hf)
"""

from __future__ import annotations
import os
import pathlib
import sys

def _setup_env() -> None:
    # Standard-Arbeitsverzeichnis
    os.environ.setdefault("WORKDIR", "/workspace")

    # Caches in /workspace/cache/hf
    cache_root = os.environ.get("CACHE_DIR", "/workspace/cache")
    os.environ.setdefault("HF_HOME", f"{cache_root}/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{cache_root}/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{cache_root}/hf")

    # Zielverzeichnis für Gewichte
    os.environ.setdefault("WEIGHTS_DIR", "/workspace/weights")

def _ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main() -> int:
    _setup_env()
    from huggingface_hub import snapshot_download

    HF_TOKEN = os.getenv("HF_TOKEN")  # optional
    HF_KW = {"token": HF_TOKEN} if HF_TOKEN else {}

    WEIGHTS_DIR = pathlib.Path(os.environ["WEIGHTS_DIR"]).resolve()
    _ensure_dir(WEIGHTS_DIR)

    # Modelle / Repos (bei Bedarf erweitern)
    targets = [
        ("Wan-AI/Wan2.1-I2V-14B-480P", WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"),
        ("MeiGen-AI/InfiniteTalk",      WEIGHTS_DIR / "InfiniteTalk"),
        ("TencentGameMate/chinese-wav2vec2-base", WEIGHTS_DIR / "chinese-wav2vec2-base"),
    ]

    print(f"[info] WEIGHTS_DIR = {WEIGHTS_DIR}")
    for repo_id, local_dir in targets:
        local_dir = local_dir.resolve()
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"[skip] {repo_id} bereits vorhanden: {local_dir}")
            continue
        print(f"[downloading] {repo_id} → {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **HF_KW,
        )

    print("\n[ok] Alle Gewichte lokal vorhanden.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
