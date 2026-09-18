# VerifAI

**Deepfake & Digital Media Authenticity Verification** — built for BRAINWAVE 2026 (ACTS EDC) by **Team SEELE**, UCER.

Upload an image, get back an authenticity score with a full evidence breakdown — not just a "real/fake" label with no reasoning.

> Live demo repo: https://github.com/SwarN1M/verifai-prototype

## What it does

VerifAI runs every uploaded image through three parallel checks and fuses them into one 0–100 authenticity score:

| Signal | What it checks |
|---|---|
| **Metadata / EXIF** | Missing camera data, stripped timestamps, known editor/AI-tool software signatures |
| **Error Level Analysis (ELA)** | Re-compresses the image and diffs it against the original to surface spliced/edited regions as a heatmap |
| **Pattern / texture check** | Frequency-domain and texture heuristics that correlate with GAN/diffusion output |

The result ships with a **verdict** (Likely Authentic / Suspicious / Likely Manipulated), a **score breakdown per signal**, a **plain-language evidence list**, and a **visual ELA heatmap** — so the answer is explainable, not a black box.

## What's real vs. a placeholder

This is a hackathon demo built to run **fully offline** — no API keys, no model downloads, no internet required to try it.

- ✅ **Metadata/EXIF check** — real, reads actual EXIF tags
- ✅ **Error Level Analysis** — real forensic technique, genuinely detects re-compression anomalies
- ✅ **Score aggregation** — real weighted fusion logic
- ⚠️ **Pattern/texture check** — a placeholder heuristic (Laplacian variance + FFT energy), not a trained deepfake classifier. Clearly marked in the UI and in `detector.py`, with the exact code to swap in a real Hugging Face model once you have internet access. See [Swapping in a real model](#swapping-in-a-real-model) below.

## Project structure

```
verifai-demo/
├── backend/
│   ├── app.py              FastAPI server (API + serves the frontend)
│   ├── detector.py         Core detection logic — metadata, ELA, pattern check, aggregator
│   ├── cli_test.py         Run the pipeline on one image from the terminal, no server needed
│   └── requirements.txt
├── frontend/
│   └── index.html          Single-file UI (upload, score, evidence, ELA heatmap)
├── sample_images/
│   ├── sample_authentic_looking.jpg
│   └── sample_manipulated_looking.jpg
├── LICENSE
└── README.md
```

## Quick start

```bash
git clone https://github.com/SwarN1M/verifai-prototype.git
cd verifai-prototype/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** and drop in an image.

### No server? Test the pipeline from the terminal

```bash
cd backend
python3 cli_test.py ../sample_images/sample_authentic_looking.jpg
```

Only needs `Pillow`, `numpy`, and `opencv-python-headless` — not FastAPI — so it's the fastest way to sanity-check the detection logic on your machine.

### About the sample images

Both sample images are synthetically generated (noise + shapes), not real photos, so they'll always score lower on the metadata check — nothing produced them with a camera, so there's no EXIF to find. To see the full range of scores, try:
- A real photo straight off your phone (should score high — EXIF, camera make/model, and natural sensor noise all check out)
- A screenshot or heavily-filtered photo (should score lower)

## Swapping in a real model

`detector.py`'s `pattern_score()` function is intentionally a placeholder. Replace it with a real Hugging Face deepfake classifier:

```bash
pip install transformers torch
```

```python
from transformers import pipeline
_clf = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

def pattern_score(image):
    results = _clf(image)
    fake_prob = next(r["score"] for r in results if r["label"].lower() == "deepfake")
    score = round(100 * (1 - fake_prob))
    flags = ["Model flagged this image as likely AI-generated."] if fake_prob > 0.5 else []
    return {"score": score, "model_raw": results, "flags": flags}
```

Nothing else in `app.py` or the frontend needs to change — the aggregator just consumes whatever `pattern_score()` returns.

## Extending to video

1. Sample ~1 frame/second from the uploaded video with `ffmpeg`/`opencv`
2. Run `analyze_image()` on each sampled frame
3. Add a frame-consistency check (face landmark / lighting drift between consecutive frames — a classic deepfake tell)
4. Average per-frame scores and add consistency as a fourth weighted signal in the aggregator

## Extending to audio

Wire in a pretrained voice-deepfake classifier the same way as the image model:

```python
from transformers import pipeline
audio_clf = pipeline("audio-classification", model="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification")
```

For video, extract the audio track first with `ffmpeg -i video.mp4 -q:a 0 -map a audio.wav`, then run it through the classifier above as a fifth weighted signal.

## Known limitations

- `pattern_score()` is a heuristic, not a trained classifier — flagged in the UI disclaimer
- ELA works best on JPEGs; PNGs (lossless) show a weaker signal since there's no compression history to diff against
- No video or audio support yet in this repo — see extension notes above
- Single-request, in-memory demo — no database, no auth, no rate limiting. Built for a hackathon judge to click through, not for production deployment

## Sources & references

- [FaceForensics++](https://justusthies.github.io/posts/faceforensics++/) — benchmark deepfake video dataset
- [Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge) — Kaggle
- [C2PA](https://c2pa.org/) — Coalition for Content Provenance & Authenticity
- [Content Credentials](https://contentcredentials.org/) — public provenance verification tool
- [Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) — Hugging Face
- [deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1) — Hugging Face

## License

MIT — see [LICENSE](LICENSE).

## Team

**SEELE** · UCER · Team Leader: Shaurya Pandey
