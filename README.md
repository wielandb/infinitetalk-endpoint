# InfiniteTalk – Inference Endpoint

Dieses Repo startet eine FastAPI mit `/generate` (Multipart Upload) und liefert das erzeugte MP4 unter `/files/{name}` aus. Low-VRAM ist als Default aktiv.

## Lokaler Test (mit GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_weights.py      # lädt WAN, InfiniteTalk, wav2vec
# Starte API
./generate.sh
# Test mit curl:
curl -X POST http://localhost:8000/generate \
  -F "prompt=Person talking" \
  -F "audio=@/path/to/voice.wav" \
  -F "ref_image=@/path/to/ref.png"
