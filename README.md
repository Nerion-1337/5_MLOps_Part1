# Prêt à dépenser — Scoring Crédit & Plateforme MLOps

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=for-the-badge&logo=astral&logoColor=white" alt="uv" />
  <img src="https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/LightGBM-Gradient%20Boosting-333333?style=for-the-badge" alt="LightGBM" />
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-1E6BBA?style=for-the-badge" alt="Optuna" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-FF6F00?style=for-the-badge" alt="SHAP" />
</p>

Ce projet met en place une chaîne MLOps de bout en bout pour la société financière « Prêt à dépenser ». L'outil prédit le risque de défaut de paiement d'un emprunteur, minimise une fonction de coût métier dissymétrique, assure la traçabilité complète des expérimentations dans MLflow et offre une interface d'aide à la décision explicable (SHAP) pour les chargés d'études.

---

## 🏗️ Architecture du Système

```mermaid
flowchart TD
    subgraph DataPrep ["1. Préparation & Traitement des Données"]
        A[Tables Kaggle Home Credit] --> B[01_exploration_nettoyage.ipynb]
        B -->|Fusion, Nettoyage, Sparsity| C[(dataset_train_fusion.parquet)]
        C --> D[03_creation_dataset.ipynb]
        D -->|Extraction & Index SK_ID_CURR| E[(X_test_sample.parquet)]
    end

    subgraph MLOpsTrain ["2. Modélisation, CV & Tracking MLflow"]
        C --> F[02_entrainement_mlflow.ipynb]
        F -->|Stratified 5-Fold CV| G[Validation Croisée Robustesse]
        F -->|Optimisation Bayésienne| H[Optuna - Coût Métier 10*FN + 1*FP]
        F -->|Seuil optimal: 0.54| I[Modèle Champion Final]
        I -->|Logs params, métriques, artefacts| J[(MLflow Tracking - metadata.db)]
        J -->|Model Registry| K[Alias @prod: CreditScoringModel]
    end

    subgraph ServingInference ["3. Déploiement Local & Inférence"]
        K -->|mlflow models serve| L[API Inférence MLflow - Port 5002]
        L <-->|POST /invocations JSON| M[test/serving_mlflow.py]
    end

    subgraph UXDashboard ["4. Aide à la Décision Métier"]
        K -->|Chargement modèle local| N[app/dashboard.py - Streamlit]
        E -->|Sélection Client par ID| N
        N --> O[Score de Risque & Recommandation]
        N --> P[Explicabilité SHAP Locale & Globale]
    end
```

---

## 📁 Structure du Projet

```text
5_MLOps_Part1/
├── data/
│   ├── raw/                         # Données brutes Kaggle (application_train, bureau...)
│   ├── processed/                   # Datasets nettoyés et échantillon de test
│   │   ├── dataset_train_fusion.parquet
│   │   └── X_test_sample.parquet    # Échantillon 50 clients indexés par SK_ID_CURR
│   └── mlflow/                      # Persistance MLflow locale (metadata.db & mlruns/)
├── notebooks/
│   ├── 01_exploration_nettoyage.ipynb  # EDA, encodage mixte, gestion des NaN et anomalies
│   ├── 02_entrainement_mlflow.ipynb    # Modélisation, 5-Fold CV, Optuna, seuil et logs
│   └── 03_creation_dataset.ipynb       # Génération de l'échantillon de test
├── app/
│   └── dashboard.py                 # Interface Streamlit d'aide à la décision et SHAP
├── test/
│   └── serving_mlflow.py            # Script d'appel HTTP sur l'endpoint /invocations
├── pyproject.toml                   # Spécification des dépendances et de l'environnement
└── README.md
```

---

## ⚙️ Installation & Prérequis

Le projet s'exécute avec Python 3.11 et utilise le gestionnaire d'environnement `uv`.

### 1. Cloner le projet et installer les dépendances

```bash
# Cloner le dépôt
git clone <URL_DU_DEPOT>
cd 5_MLOps_Part1

# Installer et synchroniser l'environnement virtuel avec uv
uv sync
```

### 2. Exécution du pipeline de données et de modélisation

Exécutez séquentiellement les notebooks dans Jupyter :

- **`notebooks/01_exploration_nettoyage.ipynb`** : ingestion, fusion et traitement des valeurs manquantes et aberrantes.
- **`notebooks/02_entrainement_mlflow.ipynb`** : validation croisée stratifiée, optimisation des hyperparamètres (Optuna), optimisation du seuil et enregistrement du champion.
- **`notebooks/03_creation_dataset.ipynb`** : création de `X_test_sample.parquet` pour l'inférence.

---

## 🚀 Déploiement Local & Utilisation

Le projet comporte trois points d'accès opérationnels.

### 1. Interface Web MLflow (Visualisation du Tracking)

Pour visualiser les runs, comparer les courbes d'apprentissage et vérifier le registre de modèles :

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow/metadata.db --port 5000
```

Accès : [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Déploiement de l'API de Serving MLflow

MLflow sert directement le modèle champion enregistré sous l'alias `prod` :

```bash
# Lancement du serveur d'inférence sur le port 5002
uv run mlflow models serve -m "models:/CreditScoringModel@prod" -p 5002 --no-conda
```

Pour tester l'endpoint d'inférence `/invocations`, exécutez dans un autre terminal :

```bash
uv run python test/serving_mlflow.py
```

### 3. Tableau de Bord d'Aide à la Décision (Streamlit)

Pour démarrer l'application interactive destinée au chargé d'études :

```bash
uv run streamlit run app/dashboard.py
```

Accès : [http://localhost:8501](http://localhost:8501)

**Fonctionnalités de l'interface :**

- **Saisie client** : sélection par identifiant bancaire réel (`SK_ID_CURR`).
- **Décision automatisée** : calcul de probabilité et arbitrage selon le seuil métier optimal (0,54).
- **Interprétabilité locale (SHAP)** : graphique en cascade (waterfall) explicitant les contributions positives et négatives des variables pour le client sélectionné.
- **Interprétabilité globale** : classement des 20 variables les plus influentes sur les décisions du modèle.