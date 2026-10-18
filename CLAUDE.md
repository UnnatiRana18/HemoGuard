# PPH Risk Prediction System — B.Tech final-year project

## Goal
Explainable, calibrated postpartum hemorrhage (PPH) risk prediction using
routinely available maternal data, plus an optional ultrasound-derived
macrosomia estimate. Prototype = FastAPI backend + static HTML/JS frontend.

## Datasets (never commit raw data; data/ is git-ignored)
1. Zhejiang minimal dataset (figshare 29186435, PLOS ONE 2025, DOI 10.1371/journal.pone.0323190)
   - data/raw/Zhejiang_dataset.xlsx, 56,014 rows, 50 cols, PPH = 2,016 (3.60%)
   - Quirks: column "Year" is actually maternal AGE. Column " history of cesarean
     section" has a leading space. "Education level" has 6 codes. Mode of delivery:
     1 = vaginal, 0 = cesarean.
   - NO institution/hospital ID exists. Do NOT do site-based validation.
     Use repeated stratified k-fold CV and report mean ± SD.
   - EXCLUDE as leakage (complications of PPH, not predictors): Uterine atony,
     Retained placenta, Placenta previa, Placental abruption, Soft birth canal
     laceration, Coagulation disorders, Uterine rupture, Maternal Near-Miss,
     embolism / Thromboembolism / Amniotic fluid embolism / Air embolism,
     and all newborn outcome columns (Apgar, birth status, discharge status).
   - No hemoglobin, previous PPH, or BMI in this dataset.
2. HC18 fetal head circumference ultrasound (Zenodo 1322001)
   - data/raw/hc18/training_set/ : 999 images *_HC.png (800x540, grayscale) +
     999 *_HC_Annotation.png ellipse outlines (outline only, must be filled).
   - data/raw/hc18/training_set_pixel_size_and_HC.csv: filename, pixel size (mm/px),
     head circumference (mm). Ignore the official test set (unlabeled).
   - Split the 999 labeled images ourselves (70/15/15, fixed seed).

## Architecture
- src/clinical/: preprocess, train (LR primary; RF and XGBoost for comparison),
  calibrate, explain (SHAP / exact log-odds decomposition for LR)
- src/imaging/: U-Net HC segmentation, mask -> ellipse -> HC in mm, HC -> macrosomia
- src/integration/pipeline.py: imaging output becomes an input feature to the
  FROZEN clinical model at inference time only
- backend/: FastAPI, loads both models once at startup, POST /api/predict,
  also serves frontend/ as static files
- frontend/: plain HTML/CSS/JS, one fetch call to /api/predict

## Honesty rules (must be reflected in code comments, README, reports)
- The two datasets are different populations; integration is pipeline-level,
  not jointly trained on linked patients.
- Validation is internal (repeated k-fold), NOT external. Never call it external.
- Risk bands are derived on training data and then frozen.
- This is a research prototype, not for clinical use.

## Conventions
- Python 3.11+, type hints, fixed random seeds, requirements.txt kept current
- Never commit data/ or large weights (use Git LFS for *.pth)
- Each module gets a small test or a runnable __main__ check
- Save metrics to outputs/*.json and plots to outputs/plots/
