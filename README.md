# Breast Cancer Classifier — Pipeline MLOps

Projet M1 IA Tuléar. Classification bénin/malin (dataset Breast Cancer Wisconsin).

**API en ligne** : https://biopsy-predict.onrender.com/docs

## Stack
sklearn + MLflow + DVC → FastAPI → Docker → GitHub → Render → Evidently AI (monitoring)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Entraîner le modèle

```bash
python src/train.py
```

## Lancer l'API en local

```bash
uvicorn app.main:app --reload
```
→ http://127.0.0.1:8000/docs

## Pipeline DVC (entraînement + monitoring)

```bash
dvc repro
```

## Exemple de requête

```bash
curl -X POST https://biopsy-predict.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```
