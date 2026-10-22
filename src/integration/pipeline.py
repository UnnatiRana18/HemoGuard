"""HemoGuard pipeline: clinical inputs (+ optional ultrasound) -> PPH risk and explanation.

The two models are NOT trained together (no public data links ultrasound to PPH). They meet
only at inference time, in the `macrosomia` input of the frozen clinical model:

  1. ultrasound image + pixel size -> U-Net -> head circumference (mm)      [or HC entered directly]
  2. HC + gestational age -> P(macrosomia)   (INTERGROWTH-21st z-score + Zhejiang base rate)
  3. P(macrosomia) replaces the yes/no macrosomia input of the clinical logistic regression.
     A linear model can take a probability directly: it gives the model's average answer
     under that uncertainty.
  4. PPH probability, Low/Moderate/High band and per-factor contributions are returned.

Fallbacks (always reported in `warnings`): if the head size is implausible for the
gestational age, the manual macrosomia answer is used; if there is none, the population
rate for that gestational age is used.

Usage:
  from pipeline import predict
  predict(clinical_dict)                              # manual macrosomia (0/1)
  predict(clinical_dict, hc_mm=345)                   # head circumference already measured
  predict(clinical_dict, image=gray_uint8, pixel_size=0.15)   # runs the U-Net

Run from the project root for a demonstration:  python src/integration/pipeline.py
"""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Clinical folder must come FIRST: both folders contain a train.py, and the clinical
# modules need their own.
sys.path.insert(0, str(ROOT / "src" / "imaging"))
sys.path.insert(0, str(ROOT / "src" / "clinical"))

import explain  # noqa: E402
import macrosomia  # noqa: E402

CLINICAL_FIELDS = ["age", "education_level", "num_pregnancies", "parity", "prenatal_checkups",
                   "prev_cesarean_count", "cesarean_history", "mode_of_delivery",
                   "gestational_age", "multiple_births", "hypertensive_disorders",
                   "cardiac_disease", "liver_disease", "gestational_diabetes"]


@lru_cache(maxsize=None)
def _bundle(variant: str) -> dict:
    return explain.load(variant)


@lru_cache(maxsize=None)
def _link() -> dict:
    return macrosomia.load_link()


@lru_cache(maxsize=None)
def _net():
    from predict import load_model  # imported lazily: needs torch and trained weights
    return load_model()


def predict(clinical: dict, image=None, pixel_size: float | None = None,
            hc_mm: float | None = None, rho: float = macrosomia.RHO_DEFAULT,
            variant: str = "full") -> dict:
    missing = [f for f in CLINICAL_FIELDS if f not in clinical]
    if missing:
        raise KeyError(f"Missing clinical fields: {missing}")
    patient = dict(clinical)
    warnings: list[str] = []
    imaging, ellipse, measured = None, None, None

    if image is not None:
        if pixel_size is None:
            raise ValueError("pixel_size (mm per pixel) is required together with an image")
        from predict import predict_hc
        net, device = _net()
        out = predict_hc(net, device, image, float(pixel_size))
        if out is None:
            warnings.append("No fetal head was found in the image.")
        else:
            measured, ellipse = out["hc_mm"], out["ellipse"]
            source = "ultrasound image"
    elif hc_mm is not None:
        measured, source = float(hc_mm), "entered head circumference"

    # decide the macrosomia input
    link = _link()
    ga = patient["gestational_age"]
    mac_source = "manual answer"
    if measured is not None:
        est = macrosomia.estimate_macrosomia(measured, ga, rho, link)
        imaging = {**est, "source": source, "ellipse": ellipse}
        if est["note"]:
            warnings.append(est["note"])
        if est["plausible"]:
            patient["macrosomia"] = est["p_macrosomia"]
            mac_source = source
        else:
            warnings.append("The image-based estimate was NOT used.")
    if mac_source == "manual answer":
        if "macrosomia" in clinical:
            patient["macrosomia"] = float(clinical["macrosomia"])
        else:
            patient["macrosomia"] = macrosomia.base_rate(ga, link)
            mac_source = "population rate for this gestational age"
            warnings.append("Macrosomia unknown: the population rate for this gestational "
                            "age was used.")

    result = explain.explain_patient(_bundle(variant), patient)
    result["macrosomia_input"] = {"source": mac_source, "value": float(patient["macrosomia"])}
    result["imaging"] = imaging
    result["warnings"] = warnings
    return result


# ----------------------------------------------------------------------------- demo
EXAMPLE = dict(age=30, education_level=2, num_pregnancies=2, parity=1, prenatal_checkups=8,
               prev_cesarean_count=0, cesarean_history=0, mode_of_delivery=1,
               gestational_age=39, multiple_births=0, hypertensive_disorders=0,
               cardiac_disease=0, liver_disease=0, gestational_diabetes=0)


def _row(label: str, r: dict) -> None:
    print(f"  {label:<44} macrosomia input {r['macrosomia_input']['value']:6.3f} | "
          f"PPH risk {r['probability']:6.2%} | {r['band']}")


def main() -> None:
    print("Example patient: 30 years, 39 weeks, vaginal delivery, no other risk factors.\n")
    print("A. Manual yes/no macrosomia answer (what the clinical model used on its own):")
    no = predict({**EXAMPLE, "macrosomia": 0})
    yes = predict({**EXAMPLE, "macrosomia": 1})
    _row("macrosomia = no", no)
    _row("macrosomia = yes", yes)
    unknown = predict(EXAMPLE)
    _row("macrosomia unknown (population rate used)", unknown)
    print(f"     warning: {unknown['warnings'][0]}")

    print("\nB. Head circumference measured on ultrasound (rho = 0.5; range for rho 0.3 - 0.7):")
    for hc in (320, 331, 345, 355):
        r = predict(EXAMPLE, hc_mm=hc)
        lo, hi = (predict(EXAMPLE, hc_mm=hc, rho=x)["probability"] for x in (0.3, 0.7))
        im = r["imaging"]
        print(f"  HC {hc} mm (z {im['z_score']:+.2f}, {im['percentile']:.0f}th pct) -> "
              f"P(macrosomia) {im['p_macrosomia']:.1%} -> PPH risk {r['probability']:.2%} "
              f"[{min(lo, hi):.2%} - {max(lo, hi):.2%}] {r['band']}")

    print("\nC. Inconsistent input (HC 150 mm at 39 weeks):")
    bad = predict({**EXAMPLE, "macrosomia": 0}, hc_mm=150)
    _row("fallback to manual answer", bad)
    for w in bad["warnings"]:
        print(f"     warning: {w}")

    print("\nD. Full pipeline on a real ultrasound image (largest head in the test split):")
    try:
        import cv2
        from dataset import load_records, make_splits
        row = make_splits(load_records())["test"].sort_values("hc_mm").iloc[-1]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        r = predict(EXAMPLE, image=img, pixel_size=float(row["pixel_size"]))
        im = r["imaging"]
        print(f"  image {row['filename']}: true HC {row['hc_mm']:.1f} mm, U-Net HC {im['hc_mm']:.1f} mm "
              f"(z {im['z_score']:+.2f}) -> P(macrosomia) {im['p_macrosomia']:.1%}")
        _row("PPH risk with image-derived macrosomia", r)
        print("  Top factors:")
        for c in r["contributions"][:4]:
            print(f"    {c['label']:<34} {c['log_odds']:+.3f}  {c['direction']}")
    except Exception as e:  # missing weights / torch / data
        print(f"  skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()