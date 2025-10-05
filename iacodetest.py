import pandas as pd
import numpy as np

# --------- CONFIGURATION ---------
# Fichiers CSV pour une étoile (à changer manuellement)
lightcurve_csv = "data/lightcurves/Kepler-22.csv"
rv_csv = "data/radial_velocity/Kepler-22_rv.csv"
exoplanet_csv = "data/exop_data/Kepler-22_exoplanets.csv"

# --------- CHARGEMENT DES DONNEES ---------
lc_df = pd.read_csv(lightcurve_csv)
rv_df = pd.read_csv(rv_csv)
exo_df = pd.read_csv(exoplanet_csv)

# --------- FONCTION DE DETECTION SIMPLE ---------
def detect_planets(lc, rv):
    """
    Détection simplifiée pour prototype hackathon.
    Retourne une liste de dicts avec les estimations principales.
    """
    # Ici on simule l'analyse IA
    # Normalement tu ferais analyse de période, transit depth, RV amplitude etc.
    
    detected = []
    
    # Exemple: si RV max > seuil et variation flux détectée → planète candidate
    rv_threshold = 1.0  # km/s simplifié
    flux_variation_threshold = 0.001  # exemple
    num_candidates = max(1, int(len(lc) / 1000))  # simule quelques détections

    for i in range(num_candidates):
        detected.append({
            "Name": "unknown",
            "Status": "candidate",
            "Period": round(np.random.uniform(5, 300), 2),
            "Radius": round(np.random.uniform(0.5, 2.5), 2),
            "Mass": round(np.random.uniform(0.5, 5.0), 2),
            "Habitable": bool(np.random.randint(0, 2)),
            # Estimations IA
            "Composition": np.random.choice(["CO2", "O2", "N2", "H2O", "CH4"]),
            "Appearance": np.random.choice(["bleu", "rouge", "gris", "vert", "jaune"]),
            "Moons_Rings": np.random.choice(["Aucune", "1 lune", "2 lunes", "anneaux"])
        })
    
    return detected

# --------- COMPARAISON AVEC LES PLANETES CONNUES ---------
def compare_with_known(detected, exo_df):
    for planet in detected:
        # Vérifie si une planète avec période proche existe déjà
        for _, row in exo_df.iterrows():
            if abs(planet["Period"] - row.get("pl_orbper", 0)) < 5:  # tolérance en jours
                planet["Name"] = row.get("pl_name", "unknown")
                planet["Status"] = "certified"
    return detected

# --------- EXECUTION ---------
detected_planets = detect_planets(lc_df, rv_df)
detected_planets = compare_with_known(detected_planets, exo_df)

# --------- AFFICHAGE ---------
for p in detected_planets:
    print(f"Name: {p['Name']}, Status: {p['Status']}, Period: {p['Period']}, "
          f"Radius: {p['Radius']}, Mass: {p['Mass']}, Habitable: {p['Habitable']}, "
          f"Composition: {p['Composition']}, Appearance: {p['Appearance']}, Moons/Rings: {p['Moons_Rings']}")

# --------- EXPORT OPTIONNEL ---------
# pd.DataFrame(detected_planets).to_csv("detected_planets.csv", index=False)
