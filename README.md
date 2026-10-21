# HemoGuard — PPH Risk Prediction System

## Overview

Explainable, calibrated postpartum hemorrhage (PPH) risk prediction using routinely available maternal data, plus an optional ultrasound-derived fetal macrosomia estimate.

## Project Structure

```
HemoGuard/
├── README.md                    # This file
├── CLAUDE.md                    # Project instructions and architecture
├── requirements.txt             # Python dependencies
├── data/                        # Raw datasets (git-ignored)
│   ├── README.md                # Data source documentation
│   ├── Zhejiang_dataset.xlsx
│   └── hc18/...
├── src/
│   ├── clinical/                # PPH risk model
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── calibrate.py
│   │   ├── explain.py
│   │   ├── dataset.py
│   │   └── model.py
│   ├── imaging/                 # HC segmentation and macrosomia
│   │   ├── macrosomia.py
│   │   └── [segmentation modules]
│   └── integration/             # Pipeline integration
│       └── pipeline.py
├── backend/                     # FastAPI server
│   ├── main.py
│   ├── predict.py
│   └── schemas.py
├── frontend/                    # Static web UI
│   └── index.html
├── models/                      # Trained models (Git LFS)
│   └── [model files]
├── outputs/
│   ├── plots/                   # Model evaluation plots
│   └── metrics.json
├── notebooks/                   # Exploratory analysis
│   └── exploration.ipynb
└── tests/                       # Unit tests
    └── test_*.py
```

## Datasets

See `data/README.md` for details on the two datasets:
1. **Zhejiang clinical dataset** — 56k maternal records with PPH outcomes
2. **HC18 ultrasound dataset** — 999 fetal head circumference images for macrosomia estimation

## Architecture

- **Clinical Model**: Logistic regression (primary), with RF and XGBoost comparisons
  - Trained on cleaned Zhejiang clinical data
  - Risk bands: Low (<20%), Medium (20-50%), High (>50%)
  - Explained via SHAP values and log-odds decomposition
  
- **Imaging Module** (optional HC-driven macrosomia):
  - U-Net to segment fetal head from HC images
  - Ellipse fitting to extract HC in mm
  - HC → macrosomia empirical relationship
  
- **Integration Pipeline**:
  - Takes HC-derived macrosomia at inference time only
  - Feeds into calibrated clinical model as an additional feature
  - Does NOT jointly train clinical model with ultrasound

- **Backend**: FastAPI serving model inference via REST API (`/api/predict`)
- **Frontend**: Static HTML/CSS/JS interface with one fetch call to backend

## UI Preview

A screenshot of the frontend interface will be added here once deployed.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/HemoGuard.git
cd HemoGuard

# Install dependencies
pip install -r requirements.txt

# If XGBoost is still missing from your active Python environment
python -m pip install xgboost

# Download datasets (see data/README.md)
# Place Zhejiang_dataset.xlsx in data/
# Extract HC18 dataset to data/hc18/

# Train models (clinical + imaging)
python -m src.clinical.train
python -m src.imaging.train_segmentation

# Run FastAPI backend
cd backend
uvicorn main:app --reload
```

If the XGBoost import fails, install it into the same Python environment used
to run training with `python -m pip install xgboost`. You can verify the
installation with `python -c "import xgboost; print(xgboost.__version__)"`.

Visit http://localhost:8000 to access the prediction interface.

## Validation

- **Internal validation**: Repeated stratified 5-fold CV on Zhejiang data
- **Risk bands**: Estimated on training data and frozen for inference
- **Population differences**: Datasets come from different populations; no paired patients

## Honesty Statement

- This is a research prototype, not for clinical use
- Explanability is provided (SHAP values, log-odds decomposition)
- Uncertainty and calibration are quantified
- No external site validation (institution-wise) because Zhejiang dataset has no hospital ID

## License

See project license file.

## Contributing

Contributions are welcome! Please follow the conventions in `CLAUDE.md`.