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
st.set_page_config(page_title="Prêt à dépenser - Scoring Crédit", layout="wide")
st.title("📊 Outil d'Aide à la Décision - Octroi de Crédit")


# --- 3. CHARGEMENT DU MODÈLE ET DES DONNÉES ---
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

    # 2. Localisation du dossier physique (Scan récursif)
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

    # Contournement du C: pour Windows avec file:///
    chemin_posix = local_model_path.resolve().as_posix()
    model_uri = f"file:///{chemin_posix}"
    
    print(f"✅ Chargement depuis l'URI : {model_uri}")
    model = mlflow.lightgbm.load_model(model_uri)

    # 3. Chargement des données
    data_path = BASE_DIR / "data" / "processed" / "X_test_sample.parquet"
    if not data_path.exists():
        st.error(f"🚨 Fichier introuvable : {data_path}")
        st.stop()

    df = pd.read_parquet(data_path)
    return model, df


# Initialisation
model, df_clients = load_model_and_data()
SEUIL_METIER = 0.54

# --- 4. BARRE LATÉRALE : SÉLECTION DU CLIENT ---
st.sidebar.header("Recherche Client")
client_id = st.sidebar.selectbox("Sélectionnez l'ID du client :", df_clients.index)

# --- 5. PRÉDICTION & INTERPRÉTABILITÉ ---
if client_id is not None:
    client_data = df_clients.loc[[client_id]].copy()

    # Nettoyage et alignement strict sur les features d'entraînement
    if "TARGET" in client_data.columns:
        client_data = client_data.drop(columns=["TARGET"])
    if hasattr(model, "feature_name_"):
        client_data = client_data[model.feature_name_]

    # Inférence
    proba_defaut = model.predict_proba(client_data)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score de Risque")
        st.metric(label="Probabilité de faillite", value=f"{proba_defaut * 100:.1f} %")
        st.write(f"*Seuil d'acceptation strict fixé à {SEUIL_METIER * 100:.0f} %*")

    with col2:
        st.subheader("Décision Recommandée")
        if proba_defaut >= SEUIL_METIER:
            st.error("❌ CRÉDIT REFUSÉ")
        else:
            st.success("✅ CRÉDIT ACCORDÉ")

    # --- 6. ONGLETS D'INTERPRÉTABILITÉ (LOCALE & GLOBALE) ---
    st.divider()
    tab_local, tab_global = st.tabs(
        ["🔍 Analyse Client (SHAP Local)", "📈 Feature Importance Globale"]
    )

    with tab_local:
        st.subheader(f"Facteurs d'influence pour le client {client_id}")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(client_data)

        fig_local, ax_local = plt.subplots(figsize=(9, 4.5))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig_local)
        plt.close(fig_local)

    with tab_global:
        st.subheader("Top 20 des variables les plus déterminantes")
        feature_names = model.feature_name_ if hasattr(model, "feature_name_") else client_data.columns
        importances = model.feature_importances_

        df_feat_imp = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values(by="Importance", ascending=False)
            .head(20)
        )

        fig_global, ax_global = plt.subplots(figsize=(9, 5.5))
        ax_global.barh(df_feat_imp["Feature"][::-1], df_feat_imp["Importance"][::-1], color="#1f77b4")
        ax_global.set_xlabel("Nombre de divisions (Splits)")
        plt.tight_layout()
        st.pyplot(fig_global)
        plt.close(fig_global)