import os, pathlib, shutil
from huggingface_hub import snapshot_download, hf_hub_download

ROOT = pathlib.Path(__file__).parent.resolve()
W = ROOT / "weights"
W.mkdir(exist_ok=True, parents=True)

HF_TOKEN = os.getenv("HF_TOKEN")  # optional, falls gated
HF_KW = {"token": HF_TOKEN} if HF_TOKEN else {}

def dl_snapshot(repo_id, local_dir):
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False, **HF_KW)

def main():
    # WAN 2.1 I2V 14B 480P (Basismodell)
    wan_dir = W / "Wan2.1-I2V-14B-480P"
    if not wan_dir.exists():
        dl_snapshot("Wan-AI/Wan2.1-I2V-14B-480P", wan_dir)

    # InfiniteTalk (spezifische Gewichte; comfyui-Unterordner enthält *.safetensors)
    it_dir = W / "InfiniteTalk"
    if not it_dir.exists():
        dl_snapshot("MeiGen-AI/InfiniteTalk", it_dir)

    # wav2vec2 (Audio-Encoder)
    w2v_dir = W / "chinese-wav2vec2-base"
    if not w2v_dir.exists():
        dl_snapshot("TencentGameMate/chinese-wav2vec2-base", w2v_dir)

    print("All weights present.")

if __name__ == "__main__":
    main()
