# Pipeline MLOps — Classification Breast Cancer

Projet M1 Intelligence Artificielle — Tuléar
Pipeline complet : **Entraînement → Code → CI/CD → Build → Deploy → Monitor**

## Architecture du pipeline

```
Data → sklearn → MLflow/DVC → model.pkl → FastAPI → GitHub (CI/CD)
                                                        │
                                                        ▼
                                                      Docker (Build)
                                                        │
                                                        ▼
                                              Hugging Face Spaces (Deploy)
                                                        │
                                                        ▼
                                              Evidently AI (Monitor) ──► boucle de réentraînement
```

## Structure du projet

```
.
├── src/train.py            # Entraînement du modèle + tracking MLflow
├── app/main.py              # API FastAPI (sert le modèle)
├── monitoring/monitor.py    # Rapport de data drift avec Evidently AI
├── Dockerfile                # Conteneurisation de l'API
├── .github/workflows/ci-cd.yml  # CI/CD : test → build Docker → deploy HF Spaces
├── requirements.txt
└── data/, models/            # Suivis par DVC (pas commit sur Git directement)
```

## 1. Installation locale

```bash
python -m venv venv
source venv/bin/activate       # (Windows : venv\Scripts\activate)
pip install -r requirements.txt
```

## 2. Entraînement (étape 1 du pipeline)

```bash
python src/train.py
```

Cela génère `data/dataset.csv` et `models/model.pkl`, et enregistre les métriques/paramètres dans MLflow.

Pour visualiser l'interface MLflow :
```bash
mlflow ui
# ouvrir http://127.0.0.1:5000
```

## 3. Initialiser DVC (versioning des données/modèle)

```bash
git init
dvc init

dvc add data/dataset.csv
dvc add models/model.pkl

git add data/dataset.csv.dvc models/model.pkl.dvc .gitignore .dvc
git commit -m "Init DVC + données et modèle versionnés"
```

Pour un stockage distant (optionnel mais recommandé pour le rendu) :
```bash
dvc remote add -d storage <url_de_votre_stockage>   # ex: Google Drive, S3...
dvc push
```

## 4. Lancer l'API en local (étape 2/3 du pipeline)

```bash
uvicorn app.main:app --reload
```

Documentation interactive : http://127.0.0.1:8000/docs

Exemple de requête :
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

## 5. Pousser sur GitHub (étape 3 : CI/CD)

```bash
git remote add origin https://github.com/<votre-user>/<votre-repo>.git
git branch -M main
git push -u origin main
```

Pour que le workflow `.github/workflows/ci-cd.yml` fonctionne entièrement, ajoutez ces **secrets** dans
GitHub → Settings → Secrets and variables → Actions :

| Secret               | Description                                      |
|-----------------------|--------------------------------------------------|
| `DOCKERHUB_USERNAME`  | Votre identifiant Docker Hub                     |
| `DOCKERHUB_TOKEN`     | Token d'accès Docker Hub                         |
| `HF_TOKEN`            | Token Hugging Face (write access)                |
| `HF_USERNAME`         | Votre nom d'utilisateur Hugging Face             |
| `HF_SPACE_NAME`       | Nom du Space créé (ex: `breast-cancer-api`)      |

> Si vous ne voulez pas configurer Docker Hub, vous pouvez simplifier le workflow
> et pousser directement vers Hugging Face Spaces (voir étape 6).

## 6. Build & Déploiement manuel (étape 4/5 : si vous ne passez pas par CI/CD)

### Build Docker en local
```bash
docker build -t breast-cancer-api .
docker run -p 7860:7860 breast-cancer-api
```

### Déployer sur Hugging Face Spaces
1. Créez un compte sur https://huggingface.co
2. Créez un nouveau **Space** → SDK = **Docker** → nom : `breast-cancer-api`
3. Clonez le repo du Space puis copiez-y les fichiers du projet (ou poussez directement) :

```bash
git remote add space https://huggingface.co/spaces/<votre-user>/breast-cancer-api
git push space main
```

4. Hugging Face build automatiquement l'image Docker et déploie l'API.
   Votre lien final ressemblera à :
   `https://huggingface.co/spaces/<votre-user>/breast-cancer-api`

## 7. Monitoring (étape 6 du pipeline)

```bash
python monitoring/monitor.py
```

Génère `monitoring/drift_report.html` : ouvrez-le dans un navigateur pour voir
le rapport de data drift et de qualité des données (Evidently AI).

En production, ce rapport doit être régénéré périodiquement avec les nouvelles
données reçues par l'API, pour détecter un data drift et déclencher un
réentraînement (retour à l'étape 1).

## Résumé pour le rendu Google Classroom

- **Capture pipeline** : capture d'écran de ce schéma + une capture de
  `/docs` (Swagger FastAPI) + une capture du rapport Evidently.
- **Repository** : lien vers votre dépôt GitHub une fois poussé.
- **Lien (Spaces Hugging Face)** : lien vers votre Space une fois déployé.
