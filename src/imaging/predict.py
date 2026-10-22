"""Measure fetal head circumference (mm) from an ultrasound image with the trained U-Net.

Pipeline for one image:
  grayscale image -> resize to 256x256 -> U-Net probability map -> resize back to the
  ORIGINAL size -> threshold 0.5 -> keep the largest blob -> fit an ellipse ->
  Ramanujan circumference in pixels -> x pixel size (mm/pixel) = head circumference in mm.

Functions for reuse (backend): load_model, predict_hc.
Run from the project root to evaluate on the 150 held-out TEST images:
  python src/imaging/predict.py
Saves: outputs/plots/hc_scatter.png
"""
import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import (IMG_SIZE, ROOT, ellipse_circumference_px, load_records, make_splits,
                     outline_to_mask)
from model import UNet

WEIGHTS_PATH = ROOT / "models" / "unet_hc18.pth"
PLOT_PATH = ROOT / "outputs" / "plots" / "hc_scatter.png"


def load_model(path=WEIGHTS_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python src/imaging/train.py")
    ckpt = torch.load(path, map_location=device)
    net = UNet(base=ckpt["base"])
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval(), device


def predict_prob(net, device, image: np.ndarray) -> np.ndarray:
    """image: grayscale uint8 at original size -> 256x256 probability map (float 0-1)."""
    small = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(net(x))[0, 0].cpu().numpy()


def clean_mask(prob: np.ndarray, size_hw: tuple[int, int]):
    """Probability map -> binary mask at the original size (largest blob only), or None."""
    h, w = size_hw
    prob_full = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob_full > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) >= 5]
    if not contours:
        return None
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return mask


def predict_hc(net, device, image: np.ndarray, pixel_size: float):
    """Return {'hc_mm', 'mask', 'ellipse'} or None if no head was found."""
    mask = clean_mask(predict_prob(net, device, image), image.shape)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    return {"hc_mm": ellipse_circumference_px(mask) * pixel_size, "mask": mask,
            "ellipse": {"center": (cx, cy), "axes": (d1, d2), "angle": angle}}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def evaluate(net, device, records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in records.iterrows():
        image = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        true_mask = outline_to_mask(cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE))
        out = predict_hc(net, device, image, float(r["pixel_size"]))
        if out is None:
            rows.append({"filename": r["filename"], "hc_true": r["hc_mm"], "hc_pred": np.nan,
                         "dice": 0.0})
            continue
        rows.append({"filename": r["filename"], "hc_true": r["hc_mm"],
                     "hc_pred": out["hc_mm"],
                     "dice": dice(out["mask"].astype(bool), true_mask.astype(bool))})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> dict:
    ok = res.dropna(subset=["hc_pred"]).copy()
    err = ok.hc_pred - ok.hc_true
    sd = err.std(ddof=1)
    return {"n": len(res), "failed": int(res.hc_pred.isna().sum()),
            "mae": err.abs().mean(), "median_abs": err.abs().median(),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias": err.mean(),
            "loa_low": err.mean() - 1.96 * sd, "loa_high": err.mean() + 1.96 * sd,
            "mape_pct": (err.abs() / ok.hc_true).mean() * 100,
            "r": float(np.corrcoef(ok.hc_true, ok.hc_pred)[0, 1]),
            "within5": (err.abs() <= 5).mean() * 100, "within10": (err.abs() <= 10).mean() * 100,
            "dice_mean": res.dice.mean(), "dice_sd": res.dice.std(ddof=1)}, ok, err


def plot(ok: pd.DataFrame, err: pd.Series, s: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    lim = [ok.hc_true.min() - 10, ok.hc_true.max() + 10]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect agreement")
    ax.scatter(ok.hc_true, ok.hc_pred, s=14, color="#0F5257", alpha=0.75)
    ax.set(xlabel="True head circumference (mm)", ylabel="Predicted head circumference (mm)",
           title=f"Predicted vs true (n={len(ok)}, r={s['r']:.3f}, MAE={s['mae']:.2f} mm)",
           xlim=lim, ylim=lim)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    mean = (ok.hc_true + ok.hc_pred) / 2
    ax.scatter(mean, err, s=14, color="#0F5257", alpha=0.75)
    for y, style, label in [(s["bias"], "-", f"Bias {s['bias']:+.2f} mm"),
                            (s["loa_high"], "--", f"+1.96 SD ({s['loa_high']:+.1f} mm)"),
                            (s["loa_low"], "--", f"-1.96 SD ({s['loa_low']:+.1f} mm)")]:
        ax.axhline(y, color="#B33A32", ls=style, lw=1, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="Mean of true and predicted (mm)", ylabel="Predicted - true (mm)",
           title="Bland-Altman agreement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def report(res: pd.DataFrame) -> None:
    s, ok, err = summarize(res)
    print(f"Images: {s['n']} | measurement failed on: {s['failed']}")
    print(f"Head circumference error (predicted - true), mm:")
    print(f"  mean absolute error {s['mae']:.2f} | median {s['median_abs']:.2f} | RMSE {s['rmse']:.2f}")
    print(f"  relative error {s['mape_pct']:.2f}% | within 5 mm: {s['within5']:.1f}% | "
          f"within 10 mm: {s['within10']:.1f}%")
    print(f"  correlation r = {s['r']:.4f}")
    print(f"  Bland-Altman: bias {s['bias']:+.2f} mm, 95% limits of agreement "
          f"[{s['loa_low']:+.2f}, {s['loa_high']:+.2f}] mm")
    print(f"  Dice at full resolution: {s['dice_mean']:.4f} +/- {s['dice_sd']:.4f}")
    thirds = pd.qcut(ok.hc_true, 3, labels=["small heads", "medium heads", "large heads"])
    print("  MAE by head size:")
    for name, g in err.abs().groupby(thirds, observed=True):
        print(f"    {name:<13} {g.mean():.2f} mm  ({ok.hc_true[g.index].min():.0f}-"
              f"{ok.hc_true[g.index].max():.0f} mm)")
    worst = err.abs().sort_values(ascending=False).head(3)
    print("  Largest errors: " + ", ".join(
        f"{ok.filename[i]} ({err[i]:+.1f} mm)" for i in worst.index))
    plot(ok, err, s)
    print("\nSaved: outputs/plots/hc_scatter.png")


def main() -> None:
    net, device = load_model()
    test = make_splits(load_records())["test"]
    print(f"Device: {device} | evaluating on the {len(test)} held-out TEST images\n")
    report(evaluate(net, device, test))


if __name__ == "__main__":
    main()