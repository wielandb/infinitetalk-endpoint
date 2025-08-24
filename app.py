#!/usr/bin/env python3
"""
app.py – FastAPI-Endpoint für InfiniteTalk
Speichert ALLES unter /workspace (persistentes Volume auf RunPod).

ENDPOINTS:
- GET  /health     → Status & Vorhandensein der Gewichte
- POST /generate   → erzeugt MP4; form-data: prompt, audio, ref_image|ref_video
- GET  /files/*    → statische Auslieferung erzeugter MP4s

ENV (optional):
- WORKDIR                         (default: /workspace)
- WEIGHTS_DIR                     (default: /workspace/weights)
- OUT_DIR                         (default: /workspace/outputs)
- CACHE_DIR                       (default: /workspace/cache)
- HF_HOME/HUGGINGFACE_HUB_CACHE/TRANSFORMERS_CACHE (default: /workspace/cache/hf)
"""

from __future__ import annotations
import os
import json
import uuid
import pathlib
import subprocess
from tempfile import TemporaryDirectory
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------- Pfade & Umgebungen erzwingen ----------------
WORKDIR = pathlib.Path(os.environ.get("WORKDIR", "/workspace")).resolve()
WEIGHTS_DIR = pathlib.Path(os.environ.get("WEIGHTS_DIR", str(WORKDIR / "weights"))).resolve()
OUT_DIR = pathlib.Path(os.environ.get("OUT_DIR", str(WORKDIR / "outputs"))).resolve()
CACHE_DIR = pathlib.Path(os.environ.get("CACHE_DIR", str(WORKDIR / "cache"))).resolve()

# Caches für HF/Transformers nach /workspace/cache/hf
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hf"))

# InfiniteTalk-Quellcode: wir erwarten das offizielle Repo hier
CODE_DIR = pathlib.Path(os.environ.get("CODE_DIR", str(WORKDIR / "InfiniteTalk"))).resolve()

WAN_DIR = WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"
IT_DIR = WEIGHTS_DIR / "InfiniteTalk"
W2V_DIR = WEIGHTS_DIR / "chinese-wav2vec2-base"

# Verzeichnisse sicherstellen
for p in (WORKDIR, WEIGHTS_DIR, OUT_DIR, CACHE_DIR, CODE_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- FastAPI-App ----------------
app = FastAPI(title="InfiniteTalk Endpoint (/workspace)", version="1.1.0")

# Statische Auslieferung erzeugter Videos
app.mount("/files", StaticFiles(directory=str(OUT_DIR)), name="files")

@app.get("/")
def root():
    return {
        "ok": True,
        "msg": "InfiniteTalk FastAPI – alles unter /workspace",
        "health": "/health",
        "generate": {"POST": "/generate"},
        "files": "/files/{name}.mp4",
        "workdir": str(WORKDIR),
        "weights_dir": str(WEIGHTS_DIR),
        "out_dir": str(OUT_DIR),
        "code_dir": str(CODE_DIR),
    }

@app.get("/health")
def health():
    status = {
        "wan": WAN_DIR.exists() and any(WAN_DIR.iterdir()),
        "infinitetalk": IT_DIR.exists() and any(IT_DIR.iterdir()),
        "wav2vec": W2V_DIR.exists() and any(W2V_DIR.iterdir()),
        "code_dir_present": CODE_DIR.exists(),
        "generate_script": (CODE_DIR / "generate_infinitetalk.py").exists(),
        "out_dir": OUT_DIR.exists(),
    }
    return {"ok": all(status.values()), "details": status}

def _run_generate(
    input_json: pathlib.Path,
    out_path: pathlib.Path,
    sample_steps: int = 40,
    mode: str = "streaming",
    low_vram: bool = True,
    extra_args: Optional[list[str]] = None,
) -> None:
    """
    Ruft das offizielle generate_infinitetalk.py im CODE_DIR auf.
    Arbeitet mit den Gewichten in /workspace/weights.
    """
    script = CODE_DIR / "generate_infinitetalk.py"
    if not script.exists():
        raise RuntimeError(f"generate_infinitetalk.py nicht gefunden unter {script}")

    cmd = [
        "python", str(script),
        "--ckpt_dir", str(WAN_DIR),
        "--infinitetalk_dir", str(IT_DIR),
        "--wav2vec_dir", str(W2V_DIR),
        "--input_json", str(input_json),
        "--mode", mode,
        "--sample_steps", str(sample_steps),
        "--save_file", str(out_path),
    ]
    if low_vram:
        cmd += ["--num_persistent_param_in_dit", "0"]
    if extra_args:
        cmd += extra_args

    env = os.environ.copy()
    # Threads klein halten (manchmal stabiler)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    # cwd = CODE_DIR (dort liegen requirements/relative Imports)
    subprocess.check_call(cmd, cwd=str(CODE_DIR), env=env)

@app.post("/generate")
async def generate(
    prompt: str = Form("A person talking"),
    mode: str = Form("streaming"),       # "clip" oder "streaming"
    sample_steps: int = Form(40),
    ref_image: UploadFile | None = File(default=None),
    ref_video: UploadFile | None = File(default=None),
    audio: UploadFile = File(...),       # Pflicht
):
    # Mindest-Checks
    if not ((WAN_DIR.exists() and any(WAN_DIR.iterdir())) and
            (IT_DIR.exists() and any(IT_DIR.iterdir())) and
            (W2V_DIR.exists() and any(W2V_DIR.iterdir())) and
            (CODE_DIR / "generate_infinitetalk.py").exists()):
        raise HTTPException(503, "Gewichte oder Code fehlen. Bitte download_weights.py ausführen und Repo prüfen.")

    if (ref_image is None) and (ref_video is None):
        raise HTTPException(400, "Bitte ref_image ODER ref_video senden.")

    # TemporaryDirectory UNTER /workspace, damit genügend Platz ist
    tmp_root = WORKDIR / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=str(tmp_root)) as td:
        tdp = pathlib.Path(td)

        # Audio speichern
        audio_path = tdp / "audio.wav"
        audio_bytes = await audio.read()
        audio_path.write_bytes(audio_bytes)

        # Referenz (Bild oder Video)
        cond_path = None
        if ref_image is not None:
            cond_path = tdp / "ref.png"
            cond_path.write_bytes(await ref_image.read())
        elif ref_video is not None:
            cond_path = tdp / "ref.mp4"
            cond_path.write_bytes(await ref_video.read())

        # Eingabe-JSON für das Script
        data = {
            "prompt": prompt,
            "cond_video": str(cond_path),
            "cond_audio": {"person1": str(audio_path)},
        }
        input_json = tdp / "input.json"
        input_json.write_text(json.dumps(data), encoding="utf-8")

        # Ziel-Dateiname in /workspace/outputs
        out_name = f"{uuid.uuid4().hex}.mp4"
        out_path = OUT_DIR / out_name

        try:
            _run_generate(
                input_json=input_json,
                out_path=out_path,
                sample_steps=int(sample_steps),
                mode=mode,
                low_vram=True,
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(500, f"Inferenz fehlgeschlagen: {e}") from e

    # Antwort
    return JSONResponse({
        "ok": True,
        "video_file": f"/files/{out_name}",
        "abs_url_hint": "Base-URL + /files/{name}",
        "params": {"mode": mode, "sample_steps": int(sample_steps)},
        "workdir": str(WORKDIR),
        "outputs_dir": str(OUT_DIR),
    })
