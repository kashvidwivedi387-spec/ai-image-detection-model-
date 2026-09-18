"""
VerifAI — core detection engine.

This module implements three independent signals and fuses them into one
authenticity score, exactly matching the "Technical Approach" flow from the
BRAINWAVE pitch deck:

    metadata & provenance check  ─┐
    error level analysis (ELA)   ─┼─▶  weighted aggregator ─▶ score + verdict
    texture / pattern heuristic  ─┘

Everything in here runs fully offline with Pillow / NumPy / OpenCV — no
model download, no API key, no internet connection required. That's
deliberate: it means the demo runs anywhere, including on hardware with no
network access.

The one signal that is a placeholder is `pattern_score()`. A real deployment
swaps that function's body for a call to a pretrained Hugging Face
deepfake/AI-image classifier (see the commented block at the bottom of this
file for the exact swap-in code). Everything else — the EXIF/metadata logic
and the ELA logic — is genuine, working forensics, not a mock.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ExifTags

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover - cv2 is optional
    _HAS_CV2 = False


# ----------------------------------------------------------------------
# 1. Metadata & provenance check
# ----------------------------------------------------------------------

SUSPICIOUS_SOFTWARE_TAGS = [
    "photoshop", "gimp", "snapseed", "faceapp", "lightroom",
    "illustrator", "picsart", "canva", "midjourney", "dall-e",
    "dalle", "stable diffusion", "reface", "facemagic",
]


def analyze_metadata(image: Image.Image) -> dict[str, Any]:
    """Reads EXIF metadata and flags signals commonly associated with
    AI-generated or edited images (missing camera data, editor signatures,
    stripped timestamps)."""
    flags: list[str] = []
    exif_raw = image.getexif()
    tags = {}
    if exif_raw:
        for tag_id, value in exif_raw.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            tags[tag_name] = value

    has_exif = len(tags) > 0
    make = tags.get("Make")
    model = tags.get("Model")
    software = str(tags.get("Software", "")).lower()
    datetime_original = tags.get("DateTimeOriginal") or tags.get("DateTime")

    if not has_exif:
        flags.append("No EXIF metadata present — common in AI-generated, "
                      "screenshotted, or re-saved images.")
    if has_exif and not (make and model):
        flags.append("EXIF present but camera make/model is missing.")
    if not datetime_original:
        flags.append("No original capture timestamp found.")
    for tag in SUSPICIOUS_SOFTWARE_TAGS:
        if tag in software:
            flags.append(f"Software tag references an editing/generation tool: '{tags.get('Software')}'.")
            break

    score = max(0, 100 - 25 * len(flags))

    return {
        "score": score,
        "has_exif": has_exif,
        "camera_make": make,
        "camera_model": model,
        "software": tags.get("Software"),
        "captured_at": str(datetime_original) if datetime_original else None,
        "flags": flags,
    }


# ----------------------------------------------------------------------
# 2. Error Level Analysis (ELA)
# ----------------------------------------------------------------------

def error_level_analysis(image: Image.Image, quality: int = 90, scale: int = 15) -> dict[str, Any]:
    """Re-compresses the image at a known JPEG quality and diffs it against
    the original. Regions that were spliced/edited re-compress differently
    than the rest of the image and light up in the diff — the classic ELA
    forensic technique."""
    rgb = image.convert("RGB")

    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    original_arr = np.asarray(rgb, dtype=np.int16)
    resaved_arr = np.asarray(resaved, dtype=np.int16)
    diff = np.abs(original_arr - resaved_arr)

    mean_diff = float(diff.mean())
    max_diff = float(diff.max())
    std_diff = float(diff.std())
    threshold = mean_diff + 2 * std_diff if std_diff > 0 else mean_diff + 1
    hotspot_ratio = float((diff.max(axis=2) > threshold).mean())

    heat = np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heat)

    anomaly = min(1.0, hotspot_ratio * 6 + (mean_diff / 40))
    score = max(0, round(100 * (1 - anomaly)))

    return {
        "score": score,
        "mean_diff": round(mean_diff, 2),
        "max_diff": round(max_diff, 2),
        "hotspot_ratio": round(hotspot_ratio, 4),
        "heatmap_image": heatmap_img,
    }


# ----------------------------------------------------------------------
# 3. Texture / frequency pattern heuristic  (placeholder for a real model)
# ----------------------------------------------------------------------

def pattern_score(image: Image.Image) -> dict[str, Any]:
    """
    DEMO-ONLY HEURISTIC — this is NOT a trained deepfake classifier.

    It looks at two cheap, classical signals that correlate with synthetic
    imagery so the pipeline has something to fuse and the demo works with
    zero downloads:

      1. Laplacian variance — GAN/diffusion output is often unnaturally
         smooth (low high-frequency detail) compared to camera noise.
      2. High-frequency FFT energy ratio — synthetic upsampling frequently
         leaves periodic / checkerboard artifacts in the frequency domain.

    These two signals are weak and easy to fool — do not present this as a
    real deepfake detector. Swap this function's body for a real model
    before using this outside a hackathon demo (see swap-in code at the
    bottom of this file).
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if _HAS_CV2:
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    else:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        pad = np.pad(gray, 1, mode="edge")
        conv = sum(
            kernel[i, j] * pad[i:i + gray.shape[0], j:j + gray.shape[1]]
            for i in range(3) for j in range(3)
        )
        laplacian_var = float(conv.var())

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    low_energy = magnitude[mask_low].sum()
    total_energy = magnitude.sum() + 1e-6
    high_freq_ratio = float(1 - (low_energy / total_energy))

    lap_component = min(1.0, laplacian_var / 800)
    freq_component = 1 - min(1.0, abs(high_freq_ratio - 0.55) / 0.4)
    natural_likelihood = (0.5 * lap_component + 0.5 * freq_component)
    score = round(100 * natural_likelihood)

    flags = []
    if lap_component < 0.25:
        flags.append("Unusually low high-frequency detail (over-smooth texture) — a common diffusion/GAN artifact.")
    if high_freq_ratio > 0.85:
        flags.append("Unusually high frequency-domain noise, atypical of natural camera sensor noise.")

    return {
        "score": score,
        "laplacian_variance": round(laplacian_var, 2),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "flags": flags,
        "note": "Heuristic placeholder — replace with a trained classifier for production use.",
    }


# ----------------------------------------------------------------------
# 4. Aggregator
# ----------------------------------------------------------------------

WEIGHTS = {"metadata": 0.25, "ela": 0.35, "pattern": 0.40}


def verdict_for(score: int) -> str:
    if score >= 70:
        return "Likely Authentic"
    if score >= 40:
        return "Suspicious — Needs Review"
    return "Likely Manipulated / AI-Generated"


@dataclass
class VerificationResult:
    score: int
    verdict: str
    metadata: dict[str, Any]
    ela: dict[str, Any]
    pattern: dict[str, Any]
    evidence: list[str] = field(default_factory=list)


def analyze_image(image: Image.Image) -> VerificationResult:
    meta = analyze_metadata(image)
    ela = error_level_analysis(image)
    pattern = pattern_score(image)

    fused = (
        WEIGHTS["metadata"] * meta["score"]
        + WEIGHTS["ela"] * ela["score"]
        + WEIGHTS["pattern"] * pattern["score"]
    )
    score = round(fused)

    evidence = list(meta["flags"]) + list(pattern["flags"])
    if ela["hotspot_ratio"] > 0.05:
        evidence.append(
            f"ELA detected localized re-compression anomalies in "
            f"{ela['hotspot_ratio'] * 100:.1f}% of the image — possible edited region."
        )
    if not evidence:
        evidence.append("No red flags found across metadata, ELA, or texture checks.")

    return VerificationResult(
        score=score,
        verdict=verdict_for(score),
        metadata=meta,
        ela=ela,
        pattern=pattern,
        evidence=evidence,
    )


# ----------------------------------------------------------------------
# Swap-in code: replace pattern_score() with a real Hugging Face model
# ----------------------------------------------------------------------
#
# from transformers import pipeline
#
# _clf = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")
#
# def pattern_score(image: Image.Image) -> dict:
#     results = _clf(image)                       # [{'label': 'Deepfake', 'score': 0.87}, ...]
#     fake_prob = next(r["score"] for r in results if r["label"].lower() == "deepfake")
#     score = round(100 * (1 - fake_prob))         # invert: higher score = more authentic
#     flags = ["Model flagged this image as likely AI-generated."] if fake_prob > 0.5 else []
#     return {"score": score, "model_raw": results, "flags": flags}
#
# This needs internet access once (to download the model) and `pip install
# transformers torch`. Nothing else in the pipeline needs to change.
