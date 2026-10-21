"""Compare Logistic Regression, Random Forest and XGBoost for PPH risk.

Repeated stratified 5-fold CV (3 repeats, seed 42) on data/processed/clinical_clean.csv.
The encoder/scaler lives inside each Pipeline, so it is fit on training folds only
(no leakage). Model settings are fixed defaults: nothing is tuned here.

Two feature sets are compared:
  full     - all 15 predictors
  no_late  - drops `macrosomia` and `mode_of_delivery`, which are only known
             close to or after delivery (macrosomia = birth weight >= 4000 g)

Run from the project root:  python src/clinical/train.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    if exc.name != "xgboost":
        raise
    import subprocess
    import sys

    print("XGBoost is not installed; installing it into the current Python environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "clinical_clean.csv"

SEED = 42
N_SPLITS, N_REPEATS = 5, 3
PREVALENCE_THRESHOLD = 0.036  # overall PPH rate in the dataset (3.6%)

TARGET = "pph"
NOT_PREDICTORS = [TARGET, "birth_weight_g"]  # birth weight is known only after birth
CATEGORICAL = ["education_level"]
NUMERIC = ["age", "num_pregnancies", "parity", "prenatal_checkups",
           "prev_cesarean_count", "gestational_age"]
LATE_KNOWN = ["macrosomia", "mode_of_delivery"]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [c for c in CATEGORICAL if c in features]
    num = [c for c in NUMERIC if c in features]
    binary = [c for c in features if c not in cat + num]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
        ("bin", "passthrough", binary),
    ])


def make_models() -> dict:
    # Fixed, untuned settings. No class weighting, so predicted values stay
    # on the probability scale (needed for Brier score and calibration later).
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
            random_state=SEED),
    }


def threshold_metrics(y, p, thr: float) -> tuple[float, float, float]:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return sens, spec, f1_score(y, pred, zero_division=0)


def fold_metrics(y, p) -> dict:
    out = {"AUROC": roc_auc_score(y, p),
           "AUPRC": average_precision_score(y, p),
           "Brier": brier_score_loss(y, p)}
    for label, thr in [("0.5", 0.5), ("0.036", PREVALENCE_THRESHOLD)]:
        sens, spec, f1 = threshold_metrics(y, p, thr)
        out[f"Sens@{label}"], out[f"Spec@{label}"], out[f"F1@{label}"] = sens, spec, f1
    return out


def cross_validate(df: pd.DataFrame, features: list[str], model) -> pd.DataFrame:
    X, y = df[features], df[TARGET].values
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    for tr, te in cv.split(X, y):
        pipe = Pipeline([("prep", make_preprocessor(features)), ("model", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        rows.append(fold_metrics(y[te], p))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    all_predictors = [c for c in df.columns if c not in NOT_PREDICTORS]
    feature_sets = {
        "full": all_predictors,
        "no_late": [c for c in all_predictors if c not in LATE_KNOWN],
    }
    print(f"Rows: {len(df)} | PPH: {int(df[TARGET].sum())} ({df[TARGET].mean():.2%}) "
          f"| CV: {N_SPLITS}-fold x {N_REPEATS} repeats = {N_SPLITS * N_REPEATS} folds")
    print(f"AUPRC of a random model = prevalence = {df[TARGET].mean():.3f}\n")

    summary = []
    for set_name, feats in feature_sets.items():
        print(f"Feature set '{set_name}': {len(feats)} predictors")
        for model_name, model in make_models().items():
            t0 = time.time()
            res = cross_validate(df, feats, model)
            row = {"features": set_name, "model": model_name}
            for col in res.columns:
                row[col] = f"{res[col].mean():.3f} ± {res[col].std():.3f}"
            summary.append(row)
            print(f"  {model_name:<19} done in {time.time() - t0:5.0f}s")

    table = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== Ranking metrics (mean ± SD over folds) ===")
    print(table[["features", "model", "AUROC", "AUPRC", "Brier"]].to_string(index=False))
    for label in ["0.5", "0.036"]:
        print(f"\n=== Threshold {label} ===")
        cols = ["features", "model", f"Sens@{label}", f"Spec@{label}", f"F1@{label}"]
        print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()