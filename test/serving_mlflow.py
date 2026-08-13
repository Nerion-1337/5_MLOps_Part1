import requests
import pandas as pd
import numpy as np

# 1. On charge ton fichier de test
df = pd.read_parquet("data/processed/X_test_sample.parquet")

# On retire la TARGET si elle est présente pour ne garder que les features
if "TARGET" in df.columns:
    df = df.drop(columns=["TARGET"])

# 2. On isole le tout premier client (1 ligne)
client_data = df.iloc[[0]].copy()
id_client = client_data.index[0]

# --- LE FIX EST ICI ---
# On remplace les 'NaN' (incompréhensibles pour JSON) par 'None' (qui deviendra 'null' en JSON)
client_data = client_data.replace({np.nan: None})
# ----------------------

# 3. On prépare les données au format strict exigé par MLflow
payload = {
    "dataframe_split": client_data.to_dict(orient="split")
}

print(f"📡 Envoi des données du client {id_client} à l'API MLflow...")

# 4. On tire sur la route /invocations !
response = requests.post(
    url="http://127.0.0.1:5002/invocations",
    json=payload
)

# 5. Affichage de la prédiction
if response.status_code == 200:
    print("\n✅ SUCCÈS ! L'API a répondu :")
    print(response.json())
else:
    print(f"\n❌ ERREUR {response.status_code} :")
    print(response.text)