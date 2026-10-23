# src/imaging/evaluate.py
import json
import os
import cv2
import numpy as np
import pandas as pd
import torch


def ramanujan_perimeter(a, b):
    """Calculates ellipse circumference using Ramanujan's approximation."""
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))


def compute_dice_and_iou(pred_mask, gt_mask):
    pred = (pred_mask > 0.5).astype(np.float32)
    gt = (gt_mask > 0.5).astype(np.float32)

    intersection = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    union = np.sum((pred + gt) > 0)

    dice = (2.0 * intersection) / (total + 1e-8) if total > 0 else 1.0
    iou = intersection / (union + 1e-8) if union > 0 else 1.0
    return float(dice), float(iou)


def extract_hc_mm(pred_mask, pixel_size):
    """Fits an ellipse to the segmented mask and calculates circumference in mm."""
    mask_u8 = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        return 0.0

    ellipse = cv2.fitEllipse(largest_contour)
    (_, _), (d1, d2), _ = ellipse
    a, b = (d1 / 2.0), (d2 / 2.0)
    perimeter_px = ramanujan_perimeter(a, b)
    return float(perimeter_px * pixel_size)


def evaluate_imaging_pipeline(
    model,
    test_loader,
    device,
    metadata_csv_path="data/raw/hc18/training_set_pixel_size_and_HC.csv",
):
    """Evaluates U-Net segmentation quality and Head Circumference error in mm."""
    model.eval()
    meta_df = pd.read_csv(metadata_csv_path).set_index("filename")

    dice_scores = []
    iou_scores = []
    hc_errors_mm = []
    hc_percentage_errors = []

    with torch.no_grad():
        for batch in test_loader:
            images, masks_gt, filenames = (
                batch["image"].to(device),
                batch["mask"],
                batch["filename"],
            )
            preds = model(images)
            preds_prob = torch.sigmoid(preds).cpu().numpy()

            for i in range(len(filenames)):
                fname = filenames[i]
                pred_mask = preds_prob[i, 0]
                gt_mask = masks_gt[i, 0].numpy()

                # Segmentation metrics
                d, iou = compute_dice_and_iou(pred_mask, gt_mask)
                dice_scores.append(d)
                iou_scores.append(iou)

                # Biometric head circumference error in mm
                if fname in meta_df.index:
                    px_size = meta_df.loc[fname, "pixel size"]
                    gt_hc = meta_df.loc[fname, "head circumference (mm)"]

                    pred_hc = extract_hc_mm(pred_mask, px_size)
                    error = abs(pred_hc - gt_hc)

                    hc_errors_mm.append(error)
                    hc_percentage_errors.append((error / gt_hc) * 100.0)

    results = {
        "Mean_Dice": float(np.mean(dice_scores)),
        "Mean_IoU": float(np.mean(iou_scores)),
        "HC_MAE_mm": float(np.mean(hc_errors_mm)),
        "HC_MAPE_percent": float(np.mean(hc_percentage_errors)),
        "HC_Max_Error_mm": float(np.max(hc_errors_mm)),
    }

    print("\n--- Imaging Model Evaluation ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/imaging_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results