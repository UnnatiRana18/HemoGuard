"""Calibrate the logistic-regression PPH model, set risk bands, freeze, evaluate once.

Order of operations (nothing is chosen using the hold-out set):
  1. Fixed stratified 80/20 split (seed 42). The 20% hold-out is untouched until step 4.
  2. On the 80% training part only: compare no calibration vs sigmoid vs isotonic using
     5-fold out-of-fold predictions. Lowest Brier wins; ties go to the simpler option.
  3. Low / Moderate / High cutoffs are the 50th and 90th percentiles of the
     out-of-fold predicted risk on the training part. They are then frozen.
  4. The frozen model and cutoffs are evaluated ONCE on the hold-out set.

Done for two feature sets: `full` (15 predictors) and `no_late` (drops macrosomia and
mode_of_delivery, which are only known near or after delivery).

Saves: models/clinical_logreg.pkl, models/clinical_logreg_no_late.pkl,
       models/risk_bands.json, outputs/plots/calibration.png
Run from the project root:  python src/clinical/calibrate.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from train import (DATA_PATH, LATE_KNOWN, NOT_PREDICTORS, ROOT, SEED, TARGET,
                   make_preprocessor)

warnings.filterwarnings("ignore")

MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "outputs" / "plots"
HOLDOUT_FRACTION = 0.20
BAND_PERCENTILES = (50, 90)  # Low < p50, Moderate p50-p90, High >= p90 of training risk
BRIER_TIE = 1e-4
MODEL_FILES = {"full": "clinical_logreg.pkl", "no_late": "clinical_logreg_no_late.pkl"}


def base_pipeline(features: list[str]) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(features)),
                     ("model", LogisticRegression(max_iter=1000))])


def candidates(features: list[str]) -> dict:
    return {
        "none": base_pipeline(features),
        "sigmoid": CalibratedClassifierCV(base_pipeline(features), method="sigmoid", cv=5),
        "isotonic": CalibratedClassifierCV(base_pipeline(features), method="isotonic", cv=5),
    }


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-frequency bins (risks are tightly clustered)."""
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, n_bins):
        total += len(idx) / len(y) * abs(p[idx].mean() - y[idx].mean())
    return total


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Regress outcome on logit(predicted risk). Ideal: slope 1, intercept 0."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e10, max_iter=1000).fit(z, y)
    return float(fit.coef_[0][0]), float(fit.intercept_[0])


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    centre = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def assign_band(p: np.ndarray, cut_low: float, cut_high: float) -> np.ndarray:
    return np.where(p < cut_low, "Low", np.where(p < cut_high, "Moderate", "High"))


def run_feature_set(name: str, features: list[str], train_df, hold_df):
    X_tr, y_tr = train_df[features], train_df[TARGET].values
    X_ho, y_ho = hold_df[features], hold_df[TARGET].values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # --- 2. choose calibration on training data only
    print(f"\n##### Feature set '{name}' ({len(features)} predictors) #####")
    print("Calibration choice (5-fold out-of-fold on the 80% training part):")
    oof, rows = {}, []
    for cname, est in candidates(features).items():
        p = cross_val_predict(est, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[cname] = p
        rows.append({"method": cname, "Brier": brier_score_loss(y_tr, p),
                     "ECE": ece(y_tr, p), "AUROC": roc_auc_score(y_tr, p)})
    table = pd.DataFrame(rows)
    print(table.round(5).to_string(index=False))
    best = table.loc[table["Brier"].idxmin(), "method"]
    if table.loc[table.method == "none", "Brier"].iloc[0] - table["Brier"].min() <= BRIER_TIE:
        best = "none"
    print(f"Selected: {best}")

    # --- 3. risk bands from training out-of-fold risk, then frozen
    cut_low, cut_high = (float(c) for c in np.percentile(oof[best], BAND_PERCENTILES))
    print(f"Frozen cutoffs: Low < {cut_low:.4f} <= Moderate < {cut_high:.4f} <= High")

    final = candidates(features)[best].fit(X_tr, y_tr)

    # --- 4. evaluate ONCE on the hold-out
    p_ho = final.predict_proba(X_ho)[:, 1]
    slope, intercept = calibration_slope_intercept(y_ho, p_ho)
    metrics = {"AUROC": roc_auc_score(y_ho, p_ho), "Brier": brier_score_loss(y_ho, p_ho),
               "ECE": ece(y_ho, p_ho), "cal_slope": slope, "cal_intercept": intercept,
               "mean_predicted": float(p_ho.mean()), "observed_rate": float(y_ho.mean())}
    print(f"\nHold-out ({len(y_ho)} rows, {int(y_ho.sum())} PPH):")
    for k, v in metrics.items():
        print(f"  {k:<15} {v:.4f}")

    bands = assign_band(p_ho, cut_low, cut_high)
    band_rows = []
    for b in ["Low", "Moderate", "High"]:
        m = bands == b
        k, n = int(y_ho[m].sum()), int(m.sum())
        lo, hi = wilson_ci(k, n)
        band_rows.append({"band": b, "n": n, "pph_cases": k, "observed_rate": k / n,
                          "ci_low": lo, "ci_high": hi, "mean_predicted": float(p_ho[m].mean())})
    bt = pd.DataFrame(band_rows)
    print("\nObserved PPH rate per band on the hold-out (95% Wilson CI):")
    for r in band_rows:
        print(f"  {r['band']:<9} n={r['n']:>5}  PPH={r['pph_cases']:>3}  "
              f"observed {r['observed_rate']:.2%} ({r['ci_low']:.2%}-{r['ci_high']:.2%})  "
              f"mean predicted {r['mean_predicted']:.2%}")
    print(f"  High vs Low relative risk: {bt.observed_rate[2] / bt.observed_rate[0]:.2f}x")

    joblib.dump(final, MODEL_DIR / MODEL_FILES[name])
    record = {"features": features, "calibration": best,
              "cutoff_low_to_moderate": cut_low, "cutoff_moderate_to_high": cut_high,
              "band_percentiles_on_training_risk": list(BAND_PERCENTILES),
              "holdout_metrics": metrics, "holdout_bands": band_rows,
              "model_file": MODEL_FILES[name]}
    return record, p_ho, y_ho, bt


def plot_all(results: dict) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 4.6 * len(results)))
    axes = np.atleast_2d(axes)
    colors = {"Low": "#2F8F5B", "Moderate": "#B9791A", "High": "#B33A32"}
    for row, (name, (rec, p, y, bt)) in enumerate(results.items()):
        ax = axes[row, 0]
        order = np.argsort(p)
        pts = [(p[i].mean(), y[i].mean()) for i in np.array_split(order, 10)]
        xs, ys = zip(*pts)
        top = max(max(xs), max(ys)) * 1.15
        ax.plot([0, top], [0, top], "k--", lw=1, label="Perfect calibration")
        ax.plot(xs, ys, "o-", color="#0F5257", label="Hold-out (10 equal-size groups)")
        ax.set(xlabel="Mean predicted PPH risk", ylabel="Observed PPH rate",
               title=f"Calibration ({name})", xlim=(0, top), ylim=(0, top))
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[row, 1]
        err = [bt.observed_rate - bt.ci_low, bt.ci_high - bt.observed_rate]
        ax.bar(bt.band, bt.observed_rate * 100, yerr=np.array(err) * 100, capsize=5,
               color=[colors[b] for b in bt.band])
        ax.axhline(y.mean() * 100, color="k", ls="--", lw=1, label="Overall rate")
        ax.set(ylabel="Observed PPH rate (%)", title=f"Risk bands on hold-out ({name})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    predictors = [c for c in df.columns if c not in NOT_PREDICTORS]
    feature_sets = {"full": predictors,
                    "no_late": [c for c in predictors if c not in LATE_KNOWN]}
    train_df, hold_df = train_test_split(df, test_size=HOLDOUT_FRACTION,
                                         stratify=df[TARGET], random_state=SEED)
    print(f"Training part: {len(train_df)} rows ({int(train_df[TARGET].sum())} PPH) | "
          f"Hold-out: {len(hold_df)} rows ({int(hold_df[TARGET].sum())} PPH)")

    results = {name: run_feature_set(name, feats, train_df, hold_df)
               for name, feats in feature_sets.items()}
    with open(MODEL_DIR / "risk_bands.json", "w") as f:
        json.dump({n: r[0] for n, r in results.items()}, f, indent=2)
    plot_all(results)
    print("\nSaved: models/clinical_logreg.pkl, models/clinical_logreg_no_late.pkl, "
          "models/risk_bands.json, outputs/plots/calibration.png")


if __name__ == "__main__":
    main()