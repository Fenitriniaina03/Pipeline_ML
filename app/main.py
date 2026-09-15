"""
API FastAPI - Étape 2/3 du pipeline.
Charge models/model.pkl et expose un endpoint de prédiction.
"""
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/model.pkl"

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="API de prédiction - Projet M1 IA Tuléar (pipeline MLOps)",
    version="1.0.0",
)

# Chargement du modèle au démarrage
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=30, max_length=30,
        description="Liste des 30 caractéristiques du dataset Breast Cancer",
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float


@app.get("/")
def root():
    return {"message": "API opérationnelle. Voir /docs pour la documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail="30 caractéristiques attendues.")

    df = pd.DataFrame([request.features], columns=FEATURE_NAMES)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][pred]

    label = "Bénin" if pred == 1 else "Malin"

    return PredictionResponse(
        prediction=int(pred),
        label=label,
        probability=float(proba),
    )
