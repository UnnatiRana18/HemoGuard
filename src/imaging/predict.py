"""Predict fetal head HC from ultrasound images using the trained U-Net.

Process (inference only):
  1. Load a test image
  2. Run the trained U-Net to predict a mask
  3. Fit an ellipse to the predicted mask (cv2.fitEllipse)
  4. Compute HC in pixels using the Ramanujan formula
  5. Convert to mm using the pixel size (correct for resize ratio)

On the test set, print:
  - Mean absolute error (MAE) in mm
  - Correlation with ground-truth HC
  - Bland-Altman summary: mean difference and limits of agreement (mean ± 1.96 SD)

Save a scatter plot predicted HC vs true HC to outputs/plots/hc_scatter.png.

Usage (from the project root):
  python src/imaging/predict.py
"""
from pathlib import Path
import subprocess
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    print("PyTorch is not installed; installing it into the current Python environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0.0"])
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True

from dataset import IMG_SIZE, read_pair, load_records, make_splits
from model import UNet

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = ROOT / "models" / "unet_hc18.pth"


def ellipse_circumference_px(mask: np.ndarray) -> float:
    """Circumference (pixels) of the ellipse fitted to a 0/1 mask (Ramanujan formula)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    (_, _), (d1, d2), _ = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    a, b = d1 / 2, d2 / 2
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def ramanujan_to_mm(circumference_px: float, pixel_size: float,
                    resize_ratio: float) -> float:
    """Convert circumference from pixels to millimetres.

    Args:
        circumference_px: HC in pixels (from fitted ellipse)
        pixel_size: Physical size of one pixel in mm (from CSV)
        resize_ratio: Ratio of original size to 256x256 (e.g., 800/256 for typical
                      HC18 images)

    Returns:
        HC in millimetres
    """
    return circumference_px * pixel_size * resize_ratio


def predict_one(net: UNet, img_processed: np.ndarray,
                 orig_shape: tuple[int, int], pixel_size: float,
                 device: torch.device) -> float:
    """Predict mask, fit ellipse, compute HC in mm for a single image.

    Args:
        net: Trained U-Net model
        img_processed: Grayscale image resized to 256x256 (0-255 uint8)
        orig_shape: (h, w) of the original image before resize (800, 540 for HC18)
        pixel_size: Physical pixel size in mm (from CSV)
        device: Torch device

    Returns:
        HC in mm (0 if mask is empty/invalid)
    """
    # Run inference
    img_tensor = torch.from_numpy(img_processed).float().div(255.0).unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H=256, W=256]
    img_tensor = img_tensor.to(device)
    net.eval()
    with torch.no_grad():
        logits = net(img_tensor)  # Expecting 4D input [B, C, H, W]
    prob = torch.sigmoid(logits).cpu().numpy().squeeze()  # [256, 256]
    mask = (prob > 0.5).astype(np.uint8)

    # Fit ellipse to predicted mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return 0.0

    (cx, cy), (d1, d2), angle = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    a, b = d1 / 2, d2 / 2
    if a <= 0 or b <= 0:
        return 0.0

    # Compute circumference in pixels
    circumference_px = float(np.pi * (3 * (a + b) -
                      np.sqrt((3 * a + b) * (a + 3 * b))))

    # Resize ratio
    resize_ratio = orig_shape[1] / IMG_SIZE  # width ratio

    # Convert to mm (correcting for resize)
    hc_mm = ramanujan_to_mm(circumference_px, pixel_size, resize_ratio)

    return hc_mm


def bland_altman(hc_pred: np.ndarray, hc_true: np.ndarray) -> dict[str, float]:
    """Compute Bland-Altman analysis.

    Args:
        hc_pred: Predicted HC values
        hc_true: Ground-truth HC values (mm)

    Returns:
        Dictionary with keys: 'mean_diff', 'sd', 'bias', 'limits_loa',
        'limits_hi_loa'
    """
    diffs = hc_pred - hc_true
    mean_diff = diffs.mean()
    sd = diffs.std()
    bias = mean_diff

    limits_loa = mean_diff - 1.96 * sd
    limits_hi_loa = mean_diff + 1.96 * sd

    return {
        "mean_diff": mean_diff,
        "sd": sd,
        "bias": bias,
        "limits_loa": limits_loa,
        "limits_hi_loa": limits_hi_loa
    }


def main() -> None:
    # Load model weights (must have been trained first)
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. Run training first:\n"
            "  python src/imaging/train.py"
        )

    # Load records and create test split
    df = load_records()
    splits = make_splits(df)
    test_records = splits["test"].copy()
    print(f"Loaded {len(test_records)} test images with labels")

    # Load trained model
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    else:
        device = torch.device("cpu")
        name = "CPU"
    model = UNet()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    print(f"Loaded UNet from {WEIGHTS_PATH.relative_to(ROOT)} (device: {device} ({name}))")

    # Predict on test set
    hc_pred_list = []
    hc_true_list = []
    failures = []

    for idx, row in test_records.iterrows():
        # Read and preprocess the image
        img_s, mask_s, orig_shape = read_pair(row, IMG_SIZE)

        # Predict mask and compute HC
        try:
            hc_pred = predict_one(model, img_s, orig_shape,
                                  row["pixel_size"], device)
            if hc_pred == 0.0:
                failures.append(idx)
            else:
                hc_pred_list.append(hc_pred)
                hc_true_list.append(row["hc_mm"])
        except Exception as e:
            failures.append(idx)
            print(f"  WARNING: Prediction failed for image {idx}: {e}")

    hc_pred = np.array(hc_pred_list) if hc_pred_list else np.array([])
    hc_true = np.array(hc_true_list) if hc_true_list else np.array([])

    print(f"\nPredictions completed: {len(hc_pred)}/{len(test_records)} successful")
    if failures:
        print(f"  Failed: {len(failures)} images")
    if hc_pred.size > 0:
        print(f"  HC range: {hc_pred.min():.1f} - {hc_pred.max():.1f} mm")
    else:
        raise RuntimeError("No predictions succeeded - check model and data")

    # Compute metrics
    mae_mm = np.abs(hc_pred - hc_true).mean()
    corr = np.corrcoef(hc_true, hc_pred)[0, 1]

    ba = bland_altman(hc_pred, hc_true)

    # Print results
    print(f"\n{'='*55}")
    print("PREDICTION PERFORMANCE ON TEST SET")
    print('='*55)
    print(f"Mean absolute error (MAE):     {mae_mm:.2f} mm")
    print(f"Correlation (pred vs true):    {corr:.4f}")
    print(f"\nBland-Altman analysis:")
    print(f"  Mean difference:            {ba['mean_diff']:+.2f} mm")
    print(f"  Standard deviation:         {ba['sd']:.2f} mm")
    print(f"  Bias:                        {ba['bias']:+.2f} mm")
    print(f"  Limits of agreement:         {ba['limits_loa']:+.2f} to "
          f"{ba['limits_hi_loa']:+.2f} mm")
    print(f"  (mean ± 1.96 × SD)")

    # Save scatter plot
    plot_path = ROOT / "outputs" / "plots" / "hc_scatter.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(hc_true, hc_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

    # Perfect prediction line
    ax.plot([hc_true.min(), hc_true.max()], [hc_true.min(), hc_true.max()],
            'r--', lw=2, label="Perfect prediction")

    # Add tolerance band
    tol = np.std(hc_pred - hc_true) * 1.96
    ax.plot([hc_true.min() - tol, hc_true.max() + tol],
            [hc_true.min() + tol, hc_true.max() + tol], 'r:', lw=1)

    ax.set_xlabel("True HC (mm)")
    ax.set_ylabel("Predicted HC (mm)")
    ax.set_title(f"HC Prediction vs Ground Truth\n"
                 f"MAE = {mae_mm:.2f} mm, r = {corr:.3f}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved scatter plot: {plot_path}")


if __name__ == "__main__":
    main()