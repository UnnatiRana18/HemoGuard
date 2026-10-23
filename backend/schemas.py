"""Request and response shapes for the HemoGuard API."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """Clinical inputs. Ranges follow the data the model was trained on (Zhejiang cohort)."""
    age: float = Field(..., ge=14, le=55, description="Maternal age, years")
    education_level: int = Field(..., ge=1, le=6, description="Dataset education code 1-6")
    num_pregnancies: int = Field(..., ge=1, le=15, description="Including this one")
    parity: int = Field(..., ge=0, le=10, description="Previous births")
    prenatal_checkups: int = Field(..., ge=0, le=30)
    prev_cesarean_count: int = Field(0, ge=0, le=6, description="Previous cesarean sections")
    mode_of_delivery: int = Field(..., ge=0, le=1, description="1 = vaginal, 0 = cesarean")
    gestational_age: float = Field(..., ge=28, le=44, description="Weeks")
    multiple_births: int = Field(0, ge=0, le=1)
    hypertensive_disorders: int = Field(0, ge=0, le=1)
    cardiac_disease: int = Field(0, ge=0, le=1)
    liver_disease: int = Field(0, ge=0, le=1)
    gestational_diabetes: int = Field(0, ge=0, le=1)
    macrosomia: Optional[float] = Field(
        None, ge=0, le=1,
        description="Known/suspected macrosomia (1/0). Leave empty if unknown; "
                    "an ultrasound or head circumference overrides it.")

    def to_clinical(self) -> dict:
        d = self.model_dump()
        d["cesarean_history"] = int(self.prev_cesarean_count > 0)
        if d["macrosomia"] is None:
            d.pop("macrosomia")
        return d


class Contribution(BaseModel):
    feature: str
    label: str
    value: Any = None
    log_odds: float
    odds_ratio_vs_average: float
    direction: str


class PredictionResponse(BaseModel):
    probability: float
    band: str
    average_patient_risk: float
    contributions: list[Contribution]
    macrosomia_input: dict[str, Any]
    imaging: Optional[dict[str, Any]] = None
    overlay_image: Optional[str] = Field(
        None, description="JPEG (base64) of the ultrasound with the fitted head outline")
    warnings: list[str] = []
    disclaimer: str = ("Research prototype built on a public dataset. Not validated for "
                       "clinical use. Does not replace clinical judgement.")