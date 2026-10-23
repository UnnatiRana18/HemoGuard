"""HemoGuard API.  Run from the project root:  uvicorn backend.main:app --reload
Then open http://localhost:8000/docs to test, and http://localhost:8000/ for the app."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend import inference
from backend.schemas import PatientInput, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
EXAMPLE_PATIENT = ('{"age": 30, "education_level": 2, "num_pregnancies": 2, "parity": 1, '
                   '"prenatal_checkups": 8, "prev_cesarean_count": 0, "mode_of_delivery": 1, '
                   '"gestational_age": 39, "multiple_births": 0, "hypertensive_disorders": 0}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_all()  # models are loaded once, not per request
    yield


app = FastAPI(title="HemoGuard PPH risk API", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "clinical_model": inference.STATE["clinical_ready"],
            "unet": inference.STATE["unet_ready"], "unet_error": inference.STATE["unet_error"]}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    patient: str = Form(..., description="Clinical inputs as a JSON string",
                        examples=[EXAMPLE_PATIENT]),
    image: UploadFile | None = File(None, description="Optional fetal head ultrasound"),
    pixel_size: float | None = Form(None, description="mm per pixel (needed with an image)"),
    hc_mm: float | None = Form(None, gt=20, lt=450,
                               description="Head circumference in mm, if already measured"),
    rho: float = Form(0.5, ge=0.1, le=0.9, description="Assumed head-size/birth-weight correlation"),
):
    try:
        clinical = PatientInput.model_validate_json(patient).to_clinical()
    except ValidationError as e:
        raise HTTPException(422, detail=[{"field": ".".join(map(str, x["loc"])), "problem": x["msg"]}
                                         for x in e.errors()])
    raw = None
    if image is not None and image.filename:
        raw = await image.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image larger than 10 MB.")
    try:
        return inference.run(clinical, raw, pixel_size, hc_mm, rho)
    except inference.ImageError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except inference.UNetUnavailable as e:
        raise HTTPException(503, str(e))


if FRONTEND_DIR.exists() and any(FRONTEND_DIR.iterdir()):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def root() -> dict:
        return {"message": "HemoGuard API is running. The web page is not built yet.",
                "docs": "/docs"}