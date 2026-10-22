"""Estimate P(macrosomia) from a measured fetal head circumference (HC) and gestational age.

Step 1 - HC z-score against a published standard. INTERGROWTH-21st fetal HC standard
  (Papageorghiou et al., Lancet 2014;384:869-79; valid 14-40 weeks; HC in mm, GA in weeks):
    mean(GA) = -28.2849 + 1.69267*GA^2 - 0.397485*GA^2*ln(GA)
    SD(GA)   = 1.98735 + 0.0136772*GA^3 - 0.00726264*GA^3*ln(GA) + 0.000976253*GA^3*ln(GA)^2
  The formulas are checked against the official centile table (38, 39, 40 weeks) when this
  file is run.

Step 2 - link to macrosomia (birth weight >= 4000 g), anchored on the Zhejiang data.
  b(GA) = observed macrosomia rate in Zhejiang for that gestational-age group.
  ASSUMPTION: birth-weight z and head z (both relative to GA) are bivariate normal with
  correlation rho. rho is NOT estimated: no public dataset links ultrasound head size to
  birth weight. Default 0.5; always report a sensitivity range (e.g. 0.3-0.7).
    z_cut = Phi^-1(1 - b(GA));  P = 1 - Phi((z_cut - rho*z_HC) / sqrt(1 - rho^2))
  By construction, the average of P over normally distributed head sizes equals b(GA).

This is a PROXY for macrosomia, not a diagnosis: head size alone says little about the
abdomen, which drives fetal weight.

Run from the project root:  python src/imaging/macrosomia.py
(refits the base rates from data/processed/clinical_clean.csv if present, and writes
models/macrosomia_link.json, which the app loads at runtime)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "clinical_clean.csv"
LINK_PATH = ROOT / "models" / "macrosomia_link.json"

GA_BINS = [("<=36", 0, 36), ("37", 37, 37), ("38", 38, 38), ("39", 39, 39),
           ("40", 40, 40), (">=41", 41, 99)]
RHO_DEFAULT = 0.5
GA_MIN, GA_MAX = 14.0, 40.0   # range of the INTERGROWTH-21st HC standard
PLAUSIBLE_Z = 4.0             # |z| above this => image and gestational age look inconsistent
MIN_RATE = 1e-4

# Official INTERGROWTH-21st HC centiles (mm): 3rd, 5th, 10th, 50th, 90th, 95th, 97th
OFFICIAL = {38: (306.1, 308.9, 313.1, 328.1, 343.0, 347.3, 350.0),
            39: (308.3, 311.2, 315.7, 331.4, 347.1, 351.5, 354.4),
            40: (309.6, 312.7, 317.4, 333.9, 350.5, 355.2, 358.3)}
OFFICIAL_Z = (-1.8808, -1.6449, -1.2816, 0.0, 1.2816, 1.6449, 1.8808)


def hc_mean(ga):
    ga = np.asarray(ga, dtype=float)
    return -28.2849 + 1.69267 * ga**2 - 0.397485 * ga**2 * np.log(ga)


def hc_sd(ga):
    ga = np.asarray(ga, dtype=float)
    lg = np.log(ga)
    return (1.98735 + 0.0136772 * ga**3 - 0.00726264 * ga**3 * lg
            + 0.000976253 * ga**3 * lg**2)


def implied_ga_weeks(hc_mm: float) -> float:
    """Gestational age at which this HC would be the median (clipped to 14-40 weeks)."""
    grid = np.linspace(GA_MIN, GA_MAX, 2601)
    return float(np.interp(hc_mm, hc_mean(grid), grid))


def fit_base_rates(data_path: Path = DATA_PATH) -> dict:
    df = pd.read_csv(data_path)
    bins = []
    for name, lo, hi in GA_BINS:
        s = df[(df.gestational_age >= lo) & (df.gestational_age <= hi)]
        bins.append({"ga_group": name, "ga_min": lo, "ga_max": hi, "n": int(len(s)),
                     "macrosomia_cases": int(s.macrosomia.sum()),
                     "rate": float(s.macrosomia.mean())})
    link = {"reference": "INTERGROWTH-21st fetal HC standard (Papageorghiou 2014)",
            "macrosomia_definition": "birth weight >= 4000 g",
            "rho_default": RHO_DEFAULT, "bins": bins,
            "source": "Zhejiang cohort, data/processed/clinical_clean.csv"}
    LINK_PATH.parent.mkdir(exist_ok=True)
    LINK_PATH.write_text(json.dumps(link, indent=2))
    return link


def load_link() -> dict:
    if not LINK_PATH.exists():
        raise FileNotFoundError(f"{LINK_PATH} not found. Run: python src/imaging/macrosomia.py")
    return json.loads(LINK_PATH.read_text())


def base_rate(ga_weeks: float, link: dict) -> float:
    g = int(round(ga_weeks))
    for b in link["bins"]:
        if b["ga_min"] <= g <= b["ga_max"]:
            return max(b["rate"], MIN_RATE)
    return max(link["bins"][-1]["rate"], MIN_RATE)


def estimate_macrosomia(hc_mm: float, ga_weeks: float, rho: float = RHO_DEFAULT,
                        link: dict | None = None) -> dict:
    link = link or load_link()
    ga_used = float(np.clip(ga_weeks, GA_MIN, GA_MAX))  # the standard stops at 40 weeks
    mean, sd = float(hc_mean(ga_used)), float(hc_sd(ga_used))
    z = (hc_mm - mean) / sd
    b = base_rate(ga_weeks, link)
    z_cut = norm.isf(b)
    p = float(norm.sf((z_cut - rho * z) / np.sqrt(1 - rho**2)))
    plausible = abs(z) <= PLAUSIBLE_Z
    note = "" if plausible else ("Head size is very far from the expected range for this "
                                 "gestational age: check the image and the gestational age.")
    if ga_weeks > GA_MAX:
        note = (note + " " if note else "") + (f"Gestational age {ga_weeks:g} > 40 weeks: "
                                               "the 40-week reference was used.")
    return {"hc_mm": float(hc_mm), "ga_weeks": float(ga_weeks), "ga_used": ga_used,
            "expected_hc_mm": mean, "z_score": float(z), "percentile": float(100 * norm.cdf(z)),
            "base_rate": b, "rho": rho, "p_macrosomia": p, "plausible": plausible,
            "implied_ga_weeks": implied_ga_weeks(hc_mm), "note": note}


def main() -> None:
    # 1. reference formulas vs official table
    worst = 0.0
    for ga, row in OFFICIAL.items():
        calc = hc_mean(ga) + np.array(OFFICIAL_Z) * hc_sd(ga)
        worst = max(worst, float(np.abs(calc - np.array(row)).max()))
    print(f"Reference check vs official INTERGROWTH-21st table (38-40 wk, 7 centiles each): "
          f"largest difference {worst:.2f} mm")
    assert worst < 1.5, "formulas do not match the official table"

    # 2. base rates
    link = fit_base_rates() if DATA_PATH.exists() else load_link()
    print("\nMacrosomia base rate by gestational age (Zhejiang):")
    for b in link["bins"]:
        print(f"  GA {b['ga_group']:>5}: n={b['n']:>6}  cases={b['macrosomia_cases']:>5}  "
              f"rate={b['rate']:.2%}")

    # 3. property check: average probability over normal head sizes equals the base rate
    zs = np.linspace(-6, 6, 4001)
    w = norm.pdf(zs) / norm.pdf(zs).sum()
    for ga in (38, 40):
        b = base_rate(ga, link)
        pm = sum(w[i] * estimate_macrosomia(hc_mean(ga) + zs[i] * hc_sd(ga), ga, link=link)["p_macrosomia"]
                 for i in range(0, len(zs), 20)) / w[::20].sum()
        print(f"Calibration-by-construction check GA {ga}: mean P = {pm:.2%} vs base rate {b:.2%}")

    # 4. examples
    print("\nExamples at 39 weeks (median HC = %.0f mm):" % hc_mean(39))
    print(f"{'HC mm':>6} {'z':>6} {'pctile':>7}   P(macrosomia) at rho = 0.3 / 0.5 / 0.7")
    for hc in (310, 331, 355, 375):
        r = [estimate_macrosomia(hc, 39, rho, link) for rho in (0.3, 0.5, 0.7)]
        print(f"{hc:>6} {r[1]['z_score']:>+6.2f} {r[1]['percentile']:>6.1f}%   "
              f"{r[0]['p_macrosomia']:.1%} / {r[1]['p_macrosomia']:.1%} / {r[2]['p_macrosomia']:.1%}")
    print("\nSame head (HC 345 mm) at different gestational ages (rho 0.5):")
    for ga in (37, 38, 39, 40, 41):
        r = estimate_macrosomia(345, ga, link=link)
        print(f"  GA {ga}: z {r['z_score']:+.2f}, P(macrosomia) {r['p_macrosomia']:.1%} "
              f"(base rate {r['base_rate']:.1%})")
    bad = estimate_macrosomia(150, 39, link=link)
    print(f"\nInconsistent input (HC 150 mm at 39 weeks): z {bad['z_score']:+.1f}, "
          f"plausible={bad['plausible']}, implied GA {bad['implied_ga_weeks']:.1f} weeks")
    print(f"  note: {bad['note']}")
    print(f"\nSaved: {LINK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()