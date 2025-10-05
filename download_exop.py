import pandas as pd
import os
import requests
from io import StringIO

# Liste des étoiles pour ton hackathon
stars = [
    "Barnards_Star", "GJ_876", "GJ_1061", "HD_40307", 
    "Kepler-22", "Proxima_Centauri", "Ross_128", 
    "Tau_Ceti", "TRAPPIST-1", "Wolf_359"
]

# URL du catalogue complet des exoplanètes NASA
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv"

print("Téléchargement du catalogue complet d'exoplanètes NASA...")
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Erreur téléchargement: {response.status_code}")

data = StringIO(response.text)
df = pd.read_csv(data)

print("Catalogue téléchargé avec succès, filtrage par étoiles...")

# Pour chaque étoile, filtrer et sauvegarder CSV
for star in stars:
    df_star = df[df['hostname'].str.replace(" ", "_").str.lower() == star.lower()]
    if df_star.empty:
        print(f"Aucune exoplanète trouvée pour {star}, CSV vide sera créé avec entêtes.")
        df_star = pd.DataFrame(columns=df.columns)
    csv_filename = f"{star}_exoplanets.csv"
    df_star.to_csv(csv_filename, index=False)
    print(f"CSV sauvegardé pour {star}: {csv_filename}")

print("Tous les CSV ont été créés dans le répertoire courant.")
