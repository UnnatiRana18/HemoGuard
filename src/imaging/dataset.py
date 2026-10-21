"""HC18 fetal head-circumference data: image + filled ellipse mask + pixel size.

* The annotation PNGs are ellipse OUTLINES, so they are filled into solid masks.
* Images and masks are resized to 256x256 (the aspect ratio changes). Keep the original
  size and pixel size from `meta`: later steps must resize a predicted mask back to the
  original 800x540 before converting pixels to millimetres.
* Split: 70/15/15 by image, seed 42, balanced across head-size deciles.
  LIMITATION: the public HC18 files carry no patient ID, so repeat scans of the same
  woman (the *_2HC, *_3HC ... files) may land in different splits. Results on the
  validation/test images can therefore be slightly optimistic. State this in the report.

Run from the project root:  python src/imaging/dataset.py
Saves: outputs/plots/hc18_preview.png
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
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    print("PyTorch is not installed; installing it into the current Python environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0.0"])
    import torch
    from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
IMG_SIZE = 256

# We look in these places (first match wins) so the script works with either layout.
IMAGE_DIRS = [ROOT / "data/raw/hc18/training_set", ROOT / "data/raw/training_set",
              ROOT / "data/raw/hc18/training_set/training_set",
              ROOT / "data/raw/training_set/training_set"]
CSV_FILES = [ROOT / "data/raw/hc18/training_set_pixel_size_and_HC.csv",
             ROOT / "data/raw/training_set_pixel_size_and_HC.csv",
             ROOT / "training_set_pixel_size_and_HC.csv"]


def _first_existing(paths: list[Path], what: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"{what} not found. Looked in:\n  " + "\n  ".join(map(str, paths)))


def load_records() -> pd.DataFrame:
    """One row per image: filename, paths, pixel_size (mm/pixel), hc_mm (ground truth)."""
    img_dir = _first_existing([d for d in IMAGE_DIRS if (d / "1_HC.png").exists()],
                              "HC18 image folder (containing 1_HC.png)")
    csv = _first_existing(CSV_FILES, "HC18 labels CSV")
    df = pd.read_csv(csv).rename(columns={"pixel size": "pixel_size",
                                          "head circumference (mm)": "hc_mm"})
    df["image_path"] = df["filename"].map(lambda f: str(img_dir / f))
    df["mask_path"] = df["filename"].map(
        lambda f: str(img_dir / f.replace("_HC.png", "_HC_Annotation.png")
                      .replace("HC.png", "HC_Annotation.png")))
    missing = [p for p in list(df.image_path) + list(df.mask_path) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files listed in the CSV are missing, "
                                f"e.g. {missing[:3]}")
    return df


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70/15/15 split, seed 42, stratified by head-size decile."""
    bins = pd.qcut(df["hc_mm"], 10, labels=False)
    train, rest = train_test_split(df, test_size=0.30, stratify=bins, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5,
                                 stratify=pd.qcut(rest["hc_mm"], 5, labels=False),
                                 random_state=SEED)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def outline_to_mask(outline: np.ndarray) -> np.ndarray:
    """Fill the ellipse outline (grayscale image) into a solid 0/1 mask, same size."""
    binary = (outline > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def read_pair(row, size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return (image uint8 size x size, mask uint8 size x size, original (h, w))."""
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    ann = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = outline_to_mask(ann)
    h, w = img.shape
    img_s = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_s, mask_s, (h, w)


class HC18Dataset(Dataset):
    """Items: image [1,256,256] float 0-1, mask [1,256,256] float 0/1, meta dict."""

    def __init__(self, records: pd.DataFrame, size: int = IMG_SIZE):
        self.records = records.reset_index(drop=True)
        pairs = [read_pair(r, size) for _, r in self.records.iterrows()]  # preload (small)
        self.images = np.stack([p[0] for p in pairs])
        self.masks = np.stack([p[1] for p in pairs])
        self.orig_shapes = [p[2] for p in pairs]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records.iloc[i]
        image = torch.from_numpy(self.images[i]).float().div(255.0).unsqueeze(0)
        mask = torch.from_numpy(self.masks[i]).float().unsqueeze(0)
        meta = {"filename": r["filename"], "pixel_size": float(r["pixel_size"]),
                "hc_mm": float(r["hc_mm"]), "orig_h": self.orig_shapes[i][0],
                "orig_w": self.orig_shapes[i][1]}
        return image, mask, meta


def ellipse_circumference_px(mask: np.ndarray) -> float:
    """Circumference (pixels) of the ellipse fitted to a 0/1 mask (Ramanujan formula)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    (_, _), (d1, d2), _ = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    a, b = d1 / 2, d2 / 2
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def ceiling_check(df: pd.DataFrame) -> pd.DataFrame:
    """Measure HC from the TRUE masks at full size and compare with the CSV value."""
    rows = []
    for _, r in df.iterrows():
        mask = outline_to_mask(cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE))
        hc_from_mask = ellipse_circumference_px(mask) * r["pixel_size"]
        rows.append({"filename": r["filename"], "hc_csv": r["hc_mm"],
                     "hc_mask": hc_from_mask, "mask_area_px": int(mask.sum())})
    out = pd.DataFrame(rows)
    out["abs_err"] = (out.hc_mask - out.hc_csv).abs()
    return out


def save_preview(ds: HC18Dataset) -> None:
    order = np.argsort(ds.records["hc_mm"].values)
    picks = [order[int(len(order) * q)] for q in (0.05, 0.5, 0.95)]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, i in enumerate(picks):
        img, mask = ds.images[i], ds.masks[i]
        overlay = np.stack([img] * 3, axis=-1).astype(float) / 255
        overlay[mask == 1] = overlay[mask == 1] * 0.55 + np.array([0.9, 0.2, 0.2]) * 0.45
        for ax, data, title in zip(axes[row], [img, mask, overlay],
                                   ["Ultrasound", "Filled mask", "Overlay"]):
            ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        axes[row, 0].text(-0.05, 0.5, f"HC {ds.records.hc_mm[i]:.0f} mm", rotation=90,
                          transform=axes[row, 0].transAxes, va="center", ha="right")
    fig.tight_layout()
    out = ROOT / "outputs/plots/hc18_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    df = load_records()
    print(f"Found {len(df)} image/mask pairs with labels.")
    splits = make_splits(df)
    for name, part in splits.items():
        print(f"  {name:<5} {len(part):>4} images | HC mean {part.hc_mm.mean():6.1f} mm "
              f"(min {part.hc_mm.min():.0f}, max {part.hc_mm.max():.0f})")

    train_ds = HC18Dataset(splits["train"])
    img, mask, meta = train_ds[0]
    print(f"\nSample item: image {tuple(img.shape)} range {img.min():.2f}-{img.max():.2f}, "
          f"mask {tuple(mask.shape)} values {sorted(set(mask.unique().tolist()))}, "
          f"original size {meta['orig_h']}x{meta['orig_w']}")
    print(f"Mask covers {mask.mean():.1%} of a 256x256 image (mean over train: "
          f"{train_ds.masks.mean():.1%})")
    save_preview(train_ds)

    chk = ceiling_check(df)
    print(f"\nCeiling check on all {len(chk)} images: HC measured from the TRUE masks "
          f"vs CSV ground truth")
    print(f"  mean abs error {chk.abs_err.mean():.2f} mm | median {chk.abs_err.median():.2f} "
          f"| 95th pct {chk.abs_err.quantile(.95):.2f} | max {chk.abs_err.max():.2f} mm")
    print(f"  correlation {np.corrcoef(chk.hc_csv, chk.hc_mask)[0, 1]:.4f}")
    bad = chk[chk.abs_err > 10]
    print(f"  images with error > 10 mm: {len(bad)}"
          + (f" e.g. {bad.filename.tolist()[:5]}" if len(bad) else ""))
    print("\nSaved: outputs/plots/hc18_preview.png")


if __name__ == "__main__":
    main()