



# Analyse Intelligente des Avis Clients — Sentiment Analysis multi-business

Projet de fin d'études (Master 2 — Expert Chef de Projet IA) : pipeline complet d'analyse NLP des avis clients, généraliste (compatible avec n'importe quelle entreprise/secteur, pas seulement la néobanque Revolut qui a servi de point de départ). Le système extrait, traite et classifie les avis en sentiment (négatif/neutre/positif) via des modèles de Machine Learning et de Deep Learning, expose les inférences via une API REST (prédiction unitaire ou par lot), et restitue les résultats sur un tableau de bord interactif.

## Objectifs

- Explorer et préparer un dataset d'avis clients multi-secteurs (scraping Play Store / App Store sur plusieurs entreprises : Revolut, Amazon, Uber, Uber Eats, Booking.com).
- Comparer une approche ML classique (TF-IDF + modèle linéaire/arbre) à une approche Deep Learning (Transformer multilingue XLM-RoBERTa).
- Cibler le sentiment (négatif/neutre/positif) plutôt que la note brute 1-5 : plus robuste et généralisable entre secteurs.
- Évaluer et interpréter les modèles (F1-score macro, matrice de confusion).
- Exposer le meilleur modèle via une API FastAPI (avis unique ou dataset complet avec score global agrégé).
- Afficher les résultats sur un dashboard.
- Conteneuriser et industrialiser la solution (Docker, CI/CD, monitoring).

## Stack technique

| Domaine | Outils |
|---|---|
| Data & Modélisation | Python 3.11, Pandas, Scikit-Learn, HuggingFace Transformers, SHAP/LIME |
| Backend / API | FastAPI, Pydantic |
| Frontend / Dashboard | Streamlit |
| MLOps & CI/CD | Docker, GitHub Actions, Trivy, Prometheus, Grafana |

## Arborescence du projet

```
.
├── data/               # /raw (données brutes), /processed (données nettoyées) — ignoré par git
├── notebooks/          # Notebooks Jupyter (EDA, expérimentation modèles)
├── src/
│   ├── data/           # Scraping, nettoyage, preprocessing
│   ├── models/         # Entraînement, évaluation, sauvegarde des poids
│   ├── api/            # Serveur FastAPI (routes, schémas Pydantic)
│   └── dashboard/      # Code du frontend
├── tests/              # Tests unitaires et d'intégration (Pytest)
├── .github/workflows/  # Pipelines CI/CD
├── infrastructure/     # Docker-compose, Prometheus, Grafana
├── requirements.txt
└── README.md
```

## Benchmark des modèles (jeu de test, 253 807 avis multi-business)

| Modèle | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Naive Bayes | 0.910 | 0.598 |
| TF-IDF + Random Forest | 0.875 | 0.595 |
| TF-IDF + Logistic Regression | 0.855 | 0.635 |
| TF-IDF + Linear SVM | 0.893 | 0.647 |
| **XLM-RoBERTa (retenu)** | **0.930** | **0.678** |

Le macro F1 (moyenne par classe) est la métrique de référence plutôt que l'accuracy seule, car le dataset reste déséquilibré (73% positif / 24% négatif / 2.8% neutre) — l'accuracy favoriserait artificiellement les modèles qui prédisent surtout "positif". XLM-RoBERTa est retenu comme modèle de production : c'est le seul à capturer le contexte (négations, nuances) plutôt qu'un simple sac de mots, et il obtient le meilleur score sur la métrique de référence.

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Avancement du workflow

- [x] Étape 1 : Initialisation (arborescence, README, .gitignore, environnement virtuel)
- [x] Étape 2 : Exploration des données (EDA)
- [x] Étape 3 : Preprocessing NLP
- [x] Étape 4 : Modélisation (5 modèles comparés, cible sentiment)
- [x] Étape 5 : Développement Backend (API FastAPI, endpoints `/predict` et `/predict/batch`)
- [x] Étape 6 : Développement Frontend (Dashboard Streamlit, dont analyse de dataset complet)
- [ ] Étape 7 : Conteneurisation & CI/CD
