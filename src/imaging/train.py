"""Train the U-Net on HC18 (image -> head mask).

Loss = BCE + Dice. Adam, lr 1e-3 (halved when validation Dice stalls), early stopping on
validation Dice. Light augmentation (horizontal flip + brightness/contrast) on training only.
The best weights (by validation Dice) are saved to models/unet_hc18.pth.

Usage (from the project root):
  python src/imaging/train.py --benchmark      # print device + estimated time, then stop
  python src/imaging/train.py                  # full training (up to 30 epochs)
  python src/imaging/train.py --epochs 2 --limit 64   # quick smoke test on 64 images
"""
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SEED, HC18Dataset, ROOT, load_records, make_splits
from model import UNet

WEIGHTS_PATH = ROOT / "models" / "unet_hc18.pth"
BATCH_SIZE = 8
LR = 1e-3
PATIENCE = 8  # epochs without validation-Dice improvement before stopping


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image Dice at threshold 0.5, averaged over the batch."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (total + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, eps: float = 1e-6):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * inter + eps) / (total + eps)).mean()
        return self.bce(logits, target) + dice_loss


def augment(img: torch.Tensor, mask: torch.Tensor):
    """Random horizontal flip + brightness/contrast jitter (training batches only)."""
    flip = torch.rand(img.size(0)) < 0.5
    img = torch.where(flip.view(-1, 1, 1, 1), img.flip(-1), img)
    mask = torch.where(flip.view(-1, 1, 1, 1), mask.flip(-1), mask)
    gain = 0.85 + 0.30 * torch.rand(img.size(0), 1, 1, 1)
    bias = 0.10 * (torch.rand(img.size(0), 1, 1, 1) - 0.5)
    return (img * gain + bias).clamp(0, 1), mask


def run_epoch(net, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    net.train(train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for img, mask, _ in loader:
            if train:
                img, mask = augment(img, mask)
            img, mask = img.to(device), mask.to(device)
            logits = net(img)
            loss = loss_fn(logits, mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = img.size(0)
            total_loss += loss.item() * b
            total_dice += dice_score(logits, mask).item() * b
            n += b
    return total_loss / n, total_dice / n


def benchmark(net, loader, loss_fn, device, n_train: int) -> None:
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    batches = [b for _, b in zip(range(6), loader)]
    times = []
    for i, (img, mask, _) in enumerate(batches):
        t0 = time.time()
        img, mask = img.to(device), mask.to(device)
        loss = loss_fn(net(img), mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i > 0:  # skip the first (warm-up) batch
            times.append(time.time() - t0)
    per_batch = float(np.mean(times))
    epoch_s = per_batch * (n_train / BATCH_SIZE) * 1.15  # +15% for validation pass
    print(f"Estimated time per epoch: {epoch_s:.0f} s  ->  30 epochs ~ {epoch_s * 30 / 60:.0f} min")
    if epoch_s > 150:
        print("That is slow. Consider training on Google Colab (free GPU) and copying "
              "models/unet_hc18.pth back.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only N train images (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({name})")

    splits = make_splits(load_records())
    train_df, val_df = splits["train"], splits["val"]
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(max(args.limit // 4, 8))
    print(f"Loading images ... (train {len(train_df)}, val {len(val_df)})")
    train_ds, val_ds = HC18Dataset(train_df), HC18Dataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    net = UNet().to(device)
    loss_fn = BCEDiceLoss()
    if args.benchmark:
        benchmark(net, train_loader, loss_fn, device, len(train_ds))
        return

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    best, best_epoch, stale = -1.0, 0, 0
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    print(f"{'epoch':>5} {'train loss':>11} {'train dice':>11} {'val loss':>9} {'val dice':>9} {'sec':>5}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = run_epoch(net, train_loader, loss_fn, device, opt)
        va_loss, va_dice = run_epoch(net, val_loader, loss_fn, device)
        sched.step(va_dice)
        mark = ""
        if va_dice > best:
            best, best_epoch, stale = va_dice, epoch, 0
            torch.save({"state_dict": net.state_dict(), "base": net.base, "img_size": 256,
                        "val_dice": va_dice, "epoch": epoch}, WEIGHTS_PATH)
            mark = "  <- saved"
        else:
            stale += 1
        print(f"{epoch:>5} {tr_loss:>11.4f} {tr_dice:>11.4f} {va_loss:>9.4f} {va_dice:>9.4f} "
              f"{time.time() - t0:>5.0f}{mark}")
        if stale >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs.")
            break
    print(f"\nBest validation Dice {best:.4f} at epoch {best_epoch}. Weights: {WEIGHTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()