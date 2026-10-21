"""Clinical preprocessing for the Zhejiang PPH dataset.

Reads data/raw/Zhejiang dataset.xlsx, drops leakage / newborn-outcome columns,
renames columns to short snake_case names, and writes
data/processed/clinical_clean.csv. No rows are dropped and nothing is imputed
(the predictors have zero missing values in this dataset).

Run from the project root:  python src/clinical/preprocess.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "Zhejiang dataset.xlsx"
OUT_PATH = ROOT / "data" / "processed" / "clinical_clean.csv"

# original column name (after stripping spaces) -> clean name
PREDICTORS = {
    "Year": "age",  # mislabelled in the source file: this is maternal age
    "Education level": "education_level",  # categorical, codes 1-6
    "Number of pregnancies": "num_pregnancies",
    "Parity": "parity",
    "Number of prenatal checkups": "prenatal_checkups",
    "Number of previous cesarean sections": "prev_cesarean_count",
    "history of cesarean section": "cesarean_history",
    "Mode of delivery": "mode_of_delivery",  # 1 = vaginal, 0 = cesarean
    "Gestational age": "gestational_age",  # weeks
    "Whether multiple births": "multiple_births",
    "hypertensive disorders of pregnancy": "hypertensive_disorders",
    "pregnancy-associated cardiac diseases": "cardiac_disease",
    "pregnancy-associated liver diseases": "liver_disease",
    "gestational diabetes": "gestational_diabetes",
    "Macrosomia": "macrosomia",  # defined as birth weight >= 4000 g
}
TARGET = {"Postpartum hemorrhage": "pph"}
# Kept only as a reference column, never a predictor (known only after birth).
REFERENCE = {"Newborn 1 weight": "birth_weight_g"}

BINARY_COLS = [
    "cesarean_history", "mode_of_delivery", "multiple_births",
    "hypertensive_disorders", "cardiac_disease", "liver_disease",
    "gestational_diabetes", "macrosomia", "pph",
]

# Dropped on purpose (documented in CLAUDE.md): complications of PPH, not predictors
LEAKAGE_COLS = [
    "Uterine rupture", "Placenta previa", "Placental abruption",
    "Soft birth canal laceration", "Uterine atony", "Retained placenta",
    "Coagulation disorders", "Maternal Near-Miss",
    "embolism", "Thromboembolism", "Amniotic fluid embolism", "Air embolism",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_clean(df: pd.DataFrame) -> pd.DataFrame:
    wanted = {**PREDICTORS, **TARGET, **REFERENCE}
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found in file: {missing}")
    # Everything not listed in `wanted` is dropped: Id, newborn outcomes, Apgar,
    # discharge status, Number of fetuses (duplicate of multiple_births),
    # Location of termination of pregnancy (unclear meaning), and LEAKAGE_COLS.
    return df[list(wanted)].rename(columns=wanted).copy()


def validate(clean: pd.DataFrame, n_raw_rows: int) -> None:
    assert len(clean) == n_raw_rows, "rows were dropped"
    assert clean.isna().sum().sum() == 0, "unexpected missing values"
    for col in BINARY_COLS:
        assert set(clean[col].unique()) <= {0, 1}, f"{col} is not 0/1"


def main() -> None:
    raw = load_raw()
    print(f"Raw file: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print("Missing values in raw predictor/target columns:",
          int(raw[list({**PREDICTORS, **TARGET})].isna().sum().sum()))
    print("Leakage columns dropped:", [c for c in LEAKAGE_COLS if c in raw.columns])

    clean = build_clean(raw)
    validate(clean, len(raw))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_PATH, index=False)

    n_pph = int(clean["pph"].sum())
    print(f"\nSaved {OUT_PATH.relative_to(ROOT)}")
    print(f"Rows: {len(clean)} | Columns: {clean.shape[1]} "
          f"({len(PREDICTORS)} predictors + target + birth_weight_g reference)")
    print(f"PPH cases: {n_pph} ({n_pph / len(clean):.2%})")
    print("\nColumn summary:")
    print(clean.describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()