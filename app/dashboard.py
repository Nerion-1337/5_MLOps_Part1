import urllib.parse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
import streamlit as st

# --- 1. DÉFINITION AUTOMATIQUE DE LA RACINE DU PROJET ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 2. CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Futurisys - Credit Scoring", layout="wide")
st.title("📊 Outil d'Aide à la Décision - Octroi de Crédit")

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model_and_data():
    db_path = BASE_DIR / "data" / "mlflow" / "metadata.db"
    sqlite_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(sqlite_uri)
    client = mlflow.tracking.MlflowClient()

    # 1. Infos du modèle en Prod
    try:
        version_prod = client.get_model_version_by_alias("CreditScoringModel", "prod")
        raw_source = version_prod.source
        run_id = version_prod.run_id
    except Exception:
        st.error("🚨 Le modèle 'CreditScoringModel' avec l'alias 'prod' est introuvable.")
        st.stop()

    clean_source = urllib.parse.unquote(raw_source).replace("\\", "/")

    # 2. Localisation du dossier physique
    local_model_path = None
    folder_ids = [run_id]
    parts = [p for p in clean_source.split("/") if p.startswith("m-") or len(p) == 32]
    folder_ids.extend(parts)

    mlmodel_files = list(BASE_DIR.rglob("MLmodel"))
    for mlm in mlmodel_files:
        str_path = str(mlm.resolve()).replace("\\", "/")
        if any(fid in str_path for fid in folder_ids if fid):
            local_model_path = mlm.parent
            break

    if local_model_path is None and mlmodel_files:
        mlmodel_files.sort(key=lambda x: x.stat().st_mtime)
        local_model_path = mlmodel_files[-1].parent

    if local_model_path is None or not local_model_path.exists():
        st.error("🚨 Impossible de trouver le dossier physique du modèle sur le disque.")
        st.stop()

    # =================================================================
    # LE FIX DÉFINITIF : URI de type fichier local (Bypass du C:)
    # =================================================================
    # On force la conversion du chemin en format Posix (C:/Users/...)
    chemin_posix = local_model_path.resolve().as_posix()
    
    # On ajoute explicitement le protocole "file:///" 
    # pour empêcher MLflow de croire que "C:" est un protocole web.
    model_uri = f"file:///{chemin_posix}"
    
    print(f"✅ Chargement depuis l'URI parfaite : {model_uri}")
    
    # MLflow comprendra enfin que c'est un dossier local
    model = mlflow.lightgbm.load_model(model_uri)

    # 3. Chargement des données
    data_path = BASE_DIR / "data" / "processed" / "X_test_sample.parquet"
    df = pd.read_parquet(data_path)

    return model, df

# Initialisation
model, df_clients = load_model_and_data()
SEUIL_METIER = 0.54

# --- 4. BARRE LATÉRALE : SÉLECTION DU CLIENT ---
st.sidebar.header("Recherche Client")
client_id = st.sidebar.selectbox("Sélectionnez l'ID du client :", df_clients.index)

# --- 5. PRÉDICTION & DÉCISION ---
if client_id:
    # On récupère la ligne du client
    client_data = df_clients.loc[[client_id]]
    
    # =========================================================
    # LE FIX DATA SCIENCE : Alignement des colonnes
    # =========================================================
    # 1. On retire manuellement TARGET si elle a été exportée par erreur
    if "TARGET" in client_data.columns:
        client_data = client_data.drop(columns=["TARGET"])
        
    # 2. Filtre ultime : on force client_data à avoir EXACTEMENT 
    # les 577 colonnes (et dans le même ordre) que lors de l'entraînement
    if hasattr(model, 'feature_name_'):
        colonnes_attendues = model.feature_name_
        client_data = client_data[colonnes_attendues]
    # =========================================================

    # Le modèle calcule la probabilité
    proba_defaut = model.predict_proba(client_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score de Risque")
        st.metric(label="Probabilité de faillite", value=f"{proba_defaut * 100:.1f} %")
        st.write(f"*Seuil d'acceptation strict fixé à {SEUIL_METIER * 100}%*")

    with col2:
        st.subheader("Décision Recommandée")
        if proba_defaut >= SEUIL_METIER:
            st.error("❌ CRÉDIT REFUSÉ")
        else:
            st.success("✅ CRÉDIT ACCORDÉ")

    # --- 6. EXPLICATION LOCALE AVEC SHAP ---
    st.divider()
    st.subheader("🔍 Explication de la décision (Feature Importance Locale)")
    st.write("Ce graphique montre l'impact des variables sur ce client précis.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(client_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)