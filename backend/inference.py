"""Model loading and prediction for the API. Models are loaded once, at startup."""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "integration"))

import pipeline  # noqa: E402

STATE = {"clinical_ready": False, "unet_ready": False, "unet_error": None}


class ImageError(ValueError):
    """The uploaded image could not be read."""


class UNetUnavailable(RuntimeError):
    """Trained U-Net weights or torch are not available."""


def load_all() -> dict:
    """Load the clinical model (required) and the U-Net (optional) once."""
    pipeline._bundle("full")
    pipeline._link()
    STATE["clinical_ready"] = True
    try:
        pipeline._net()
        STATE["unet_ready"] = True
    except Exception as e:  # missing torch or weights: clinical-only mode still works
        STATE["unet_error"] = f"{type(e).__name__}: {e}"
    return STATE


def decode_image(raw: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ImageError("The file is not a readable image (use PNG or JPEG).")
    if min(arr.shape) < 64:
        raise ImageError("The image is too small.")
    return arr


def make_overlay(gray: np.ndarray, ellipse: dict, hc_mm: float) -> str:
    """Ultrasound with the fitted head outline, as a base64 JPEG (max width 800 px)."""
    scale = min(1.0, 800 / gray.shape[1])
    img = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
    (cx, cy), (d1, d2) = ellipse["center"], ellipse["axes"]
    cv2.ellipse(img, (int(cx * scale), int(cy * scale)),
                (max(int(d1 * scale / 2), 1), max(int(d2 * scale / 2), 1)),
                ellipse["angle"], 0, 360, (80, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"HC {hc_mm:.0f} mm", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (80, 200, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def run(clinical: dict, image_bytes: bytes | None = None, pixel_size: float | None = None,
        hc_mm: float | None = None, rho: float = 0.5) -> dict:
    gray = None
    if image_bytes is not None:
        if pixel_size is None or pixel_size <= 0:
            raise ValueError("pixel_size (mm per pixel) is required with an ultrasound image. "
                             "Alternatively enter the head circumference in mm directly.")
        if not STATE["unet_ready"]:
            raise UNetUnavailable("Image measurement is not available on this server "
                                  f"({STATE['unet_error']}). Enter the head circumference "
                                  "in mm instead.")
        gray = decode_image(image_bytes)
    result = pipeline.predict(clinical, image=gray, pixel_size=pixel_size, hc_mm=hc_mm, rho=rho)
    imaging = result.get("imaging")
    result["overlay_image"] = None
    if gray is not None and imaging and imaging.get("ellipse"):
        result["overlay_image"] = make_overlay(gray, imaging["ellipse"], imaging["hc_mm"])
    return result