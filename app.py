import os, json, uuid, pathlib, subprocess, shutil
from tempfile import TemporaryDirectory
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Pfade ---
ROOT = pathlib.Path(__file__).parent.resolve()
WEIGHTS_DIR = ROOT / "weights"
WAN_DIR = WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"
IT_DIR = WEIGHTS_DIR / "InfiniteTalk"
W2V_DIR = WEIGHTS_DIR / "chinese-wav2vec2-base"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# --- App ---
app = FastAPI(title="InfiniteTalk Endpoint", version="1.0.0")
app.mount("/files", StaticFiles(directory=str(OUT_DIR)), name="files")

# --- Health ---
@app.get("/health")
def health():
    ok = WAN_DIR.exists() and IT_DIR.exists() and W2V_DIR.exists()
    return {"status":"ok" if ok else "missing-weights",
            "wan": WAN_DIR.exists(), "infinitetalk": IT_DIR.exists(), "wav2vec": W2V_DIR.exists()}

def _run_generate(input_json: pathlib.Path, out_path: pathlib.Path,
                  sample_steps: int = 40,
                  mode: str = "streaming",
                  low_vram: bool = True,
                  extra_args: list[str] | None = None):
    cmd = [
        "python","generate_infinitetalk.py",
        "--ckpt_dir", str(WAN_DIR),
        "--infinitetalk_dir", str(IT_DIR),
        "--wav2vec_dir", str(W2V_DIR),
        "--input_json", str(input_json),
        "--mode", mode,
        "--sample_steps", str(sample_steps),
        "--save_file", str(out_path),
    ]
    if low_vram:
        cmd += ["--num_persistent_param_in_dit","0"]
    if extra_args:
        cmd += extra_args

    env = os.environ.copy()
    # leichte Stabilität: keine MKL-Threads fluten
    env.setdefault("OMP_NUM_THREADS","1")
    env.setdefault("MKL_NUM_THREADS","1")

    subprocess.check_call(cmd, cwd=str(ROOT), env=env)

@app.post("/generate")
async def generate(
    prompt: str = Form("A person speaking"),
    mode: str = Form("streaming"),     # "clip" oder "streaming"
    sample_steps: int = Form(40),
    ref_image: UploadFile | None = File(default=None),
    ref_video: UploadFile | None = File(default=None),
    audio: UploadFile = File(...),     # Pflicht
):
    if not (WAN_DIR.exists() and IT_DIR.exists() and W2V_DIR.exists()):
        raise HTTPException(503, "Weights not present. Run weight download first.")

    if (ref_image is None) and (ref_video is None):
        raise HTTPException(400, "Provide ref_image or ref_video.")

    with TemporaryDirectory() as td:
        td = pathlib.Path(td)
        # Dateien speichern
        audio_path = td / "audio.wav"
        audio_bytes = await audio.read()
        audio_path.write_bytes(audio_bytes)

        cond_path = None
        if ref_image is not None:
            cond_path = td / "ref.png"
            cond_path.write_bytes(await ref_image.read())
        elif ref_video is not None:
            cond_path = td / "ref.mp4"
            cond_path.write_bytes(await ref_video.read())

        # Input-JSON bauen
        data = {
            "prompt": prompt,
            "cond_video": str(cond_path),
            "cond_audio": {"person1": str(audio_path)}
        }
        input_json = td / "input.json"
        input_json.write_text(json.dumps(data), encoding="utf-8")

        # Output-Ziel
        out_name = f"{uuid.uuid4().hex}.mp4"
        out_path = OUT_DIR / out_name

        try:
            _run_generate(input_json, out_path, sample_steps=sample_steps, mode=mode, low_vram=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(500, f"Inference failed: {e}") from e

    return JSONResponse({
        "ok": True,
        "video_file": f"/files/{out_name}",
        "abs_url_hint": "Base-URL + /files/{name} (je nach Endpoint-Domain)",
        "params": {"mode": mode, "sample_steps": sample_steps}
    })
