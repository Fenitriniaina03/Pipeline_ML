"""
Étape 6 du pipeline : Monitoring avec Evidently AI.
Compare les données de référence (entraînement) aux nouvelles données
reçues en production pour détecter un data drift.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

REFERENCE_PATH = "data/dataset.csv"
REPORT_PATH = "monitoring/drift_report.html"


def simulate_production_data(df: pd.DataFrame, n=100, noise=0.15) -> pd.DataFrame:
    """Simule de nouvelles données de production avec un léger bruit,
    pour illustrer un data drift dans la démo."""
    sample = df.sample(n=n, replace=True, random_state=1).drop(columns=["target"]).copy()
    numeric_cols = sample.columns
    sample[numeric_cols] = sample[numeric_cols] * (1 + noise)
    return sample


def run_monitoring():
    reference = pd.read_csv(REFERENCE_PATH)
    current = simulate_production_data(reference)

    reference_features = reference.drop(columns=["target"])

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference_features, current_data=current)
    report.save_html(REPORT_PATH)
    print(f"Rapport de monitoring généré : {REPORT_PATH}")


if __name__ == "__main__":
    run_monitoring()
