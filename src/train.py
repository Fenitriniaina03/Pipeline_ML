"""
Script d'entraînement du modèle.
Étape 1 du pipeline : Entrainement (sklearn) + tracking MLflow + sauvegarde en pickle.
"""
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"


def load_data():
    """Charge le dataset (Breast Cancer Wisconsin) et le sauvegarde en CSV pour DVC."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # features + colonne 'target'
    df.to_csv(DATA_PATH, index=False)
    return df


def train():
    mlflow.set_experiment("breast-cancer-classifier")

    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "random_state": 42,
    }

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

        # Log MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("Métriques du modèle :", metrics)

        # Sauvegarde du modèle en .pkl pour l'API FastAPI (étape 2 du pipeline)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        print(f"Modèle sauvegardé dans {MODEL_PATH}")

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train()
