import streamlit as st
import pandas as pd
import shap
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Futurisys - Credit Scoring", layout="wide")
st.title("📊 Outil d'Aide à la Décision - Octroi de Crédit")

# --- 2. DÉFINITION DE LA RACINE DU PROJET ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 3. CHARGEMENT DU MODÈLE ET DES DONNÉES ---
@st.cache_resource
def load_model_and_data():
    # 1. On pointe directement vers le dossier physique du Model Registry local
    registry_dir = BASE_DIR / "data" / "mlflow" / "mlruns" / "models" / "CreditScoringModel"
    
    # 2. On scanne pour trouver tous les fichiers MLmodel
    fichiers_modeles = list(registry_dir.rglob("MLmodel"))
    
    if not fichiers_modeles:
        st.error(f"🚨 Aucun modèle trouvé dans {registry_dir}")
        st.stop()
        
    # On trie par date de modification et on garde la version la plus récente !
    fichiers_modeles.sort(key=lambda x: x.stat().st_mtime)
    chemin_modele = fichiers_modeles[-1].parent
    
    print(f"✅ BINGO ! Modèle chargé depuis : {chemin_modele}")
    
    # 3. Chargement du modèle
    model = mlflow.lightgbm.load_model(str(chemin_modele))
    
    # 4. Chargement de l'échantillon de données
    data_path = BASE_DIR / "data" / "processed" / "X_test_sample.parquet"
    df = pd.read_parquet(data_path)
    
    return model, df

model, df_clients = load_model_and_data()
SEUIL_METIER = 0.54

# --- 4. BARRE LATÉRALE : SÉLECTION DU CLIENT ---
st.sidebar.header("Recherche Client")
client_id = st.sidebar.selectbox("Sélectionnez l'ID du client :", df_clients.index)

# --- 5. PRÉDICTION & DÉCISION ---
if client_id:
    # On récupère la ligne de données du client
    client_data = df_clients.loc[[client_id]]
    
    # Le modèle calcule la probabilité de défaut (Classe 1)
    proba_defaut = model.predict_proba(client_data)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score de Risque")
        st.metric(label="Probabilité de faillite", value=f"{proba_defaut*100:.1f} %")
        st.write(f"*Seuil d'acceptation strict fixé à {SEUIL_METIER*100}%*")
        
    with col2:
        st.subheader("Décision Recommandée")
        if proba_defaut >= SEUIL_METIER:
            st.error("❌ CRÉDIT REFUSÉ")
        else:
            st.success("✅ CRÉDIT ACCORDÉ")

    # --- 6. EXPLICATION LOCALE AVEC SHAP ---
    st.divider()
    st.subheader("🔍 Explication de la décision (Feature Importance Locale)")
    st.write("Ce graphique montre pourquoi ce score précis a été attribué à ce client.")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(client_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)