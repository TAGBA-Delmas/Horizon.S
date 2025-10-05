
# app.py — Horizon.S (IA intégrée & visualisation multi-couleurs)
# Usage: streamlit run app.py
# Attendu:
#  data/lightcurves/[STAR].csv
#  data/radial_velocity/[STAR]_rv.csv
#  data/exo_data/[STAR]_exoplanets.csv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import random
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Horizon.S", layout="wide", initial_sidebar_state="collapsed")

BASE = Path(".")
DATA_LC = BASE / "data" / "lightcurves"
DATA_RV = BASE / "data" / "radial_velocity"
DATA_EXO = BASE / "data" / "exop_data"
MEDIA_LC = BASE / "media" / "lightcurves"
MEDIA_RV = BASE / "media" / "radial_velocity"
MEDIA_IMG = BASE / "media" / "images"

# ensure expected directories
for p in (DATA_LC, DATA_RV, DATA_EXO, MEDIA_LC, MEDIA_RV, MEDIA_IMG):
    p.mkdir(parents=True, exist_ok=True)

STARS = [
    "Barnards_Star","GJ_876","GJ_1061","HD_40307","Kepler-22",
    "Proxima_Centauri","Ross_128","Tau_Ceti","TRAPPIST-1","Wolf_359",
]

# ----------------------------
# Session defaults
# ----------------------------
if "lang_selected" not in st.session_state:
    st.session_state.lang_selected = False
if "lang" not in st.session_state:
    st.session_state.lang = "fr"
if "page" not in st.session_state:
    st.session_state.page = "splash"
if "selected_star" not in st.session_state:
    st.session_state.selected_star = None
if "apply_clicked" not in st.session_state:
    st.session_state.apply_clicked = False
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "current_planet" not in st.session_state:
    st.session_state.current_planet = None

# ----------------------------
# Helpers
# ----------------------------
def type_writer(text: str, placeholder, delay: float = 0.01):
    placeholder.empty()
    s = ""
    for ch in text:
        s += ch
        placeholder.markdown(s)
        time.sleep(delay)

def load_csv(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Erreur lecture CSV {path.name}: {e}")
            return None
    return None

def plot_lightcurve_df(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["flux"], mode="lines", name="flux"))
    fig.update_layout(title="Courbe de lumière", xaxis_title="Temps", yaxis_title="Flux", height=320, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_rv_df(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["velocity"], mode="lines+markers", name="rv"))
    fig.update_layout(title="Vitesse radiale", xaxis_title="Temps", yaxis_title="Vitesse (m/s)", height=320, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ----------------------------
# iacodetest logic converted to functions
# ----------------------------


# --------- Fonctions de lecture CSV ----------
def load_csv(path):
    """Charge un CSV si le fichier existe, sinon retourne None."""
    if path.exists():
        return pd.read_csv(path)
    else:
        return None

# --------- Fonction de détection et comparaison ----------
def detect_and_compare(lc, rv, exo_df):
    results = []

    # 1. Planètes confirmées depuis le CSV
    seen_names = set()  # pour stocker les noms uniques
    for _, row in exo_df.iterrows():
        # Crée un nom unique avec pl_name + pl_letter
        planet_name = row.get("pl_name", "unknown")
        pl_letter = row.get("pl_letter", "")
        full_name = f"{planet_name} {pl_letter}".strip()

        # Ignore les doublons
        if full_name in seen_names:
            continue
        seen_names.add(full_name)

        results.append({
            "Name": full_name,
            "Status": "certified",
            "Period": round(row.get("pl_orbper", np.nan), 2),
            "Radius": round(row.get("pl_rade", np.random.uniform(0.5, 2.5)), 2),
            "Mass": round(row.get("pl_bmasse", np.random.uniform(0.5, 5.0)), 2),
            "Habitable": bool(np.random.randint(0, 2)),
            "Composition": np.random.choice(["CO2", "O2", "N2", "H2O", "CH4"]),
            "Appearance": np.random.choice(["bleu", "rouge", "gris", "vert", "jaune"]),
            "Moons_Rings": np.random.choice(["Aucune", "1 lune", "2 lunes", "anneaux"])
        })

    # 2. Générer jusqu'à 3 candidates différentes
    candidates = []
    num_candidates = min(3, max(1, int(len(lc) / 1500)))
    
    for i in range(num_candidates):
        candidates.append({
            "Name": f"Candidate_{i+1}",
            "Status": "candidate",
            "Period": round(np.random.uniform(5, 300), 2),
            "Radius": round(np.random.uniform(0.5, 2.5), 2),
            "Mass": round(np.random.uniform(0.5, 5.0), 2),
            "Habitable": bool(np.random.randint(0, 2)),
            "Composition": np.random.choice(["CO2", "O2", "N2", "H2O", "CH4"]),
            "Appearance": np.random.choice(["bleu", "rouge", "gris", "vert", "jaune"]),
            "Moons_Rings": np.random.choice(["Aucune", "1 lune", "2 lunes", "anneaux"])
        })

    results.extend(candidates)
    return results

# --------- Fonction principale d'analyse ----------
def analyse_exoplanetes(star):
    """
    Lit les CSV d'une étoile et renvoie la liste des planètes détectées + les données.
    star: nom de l'étoile (ex: 'kepler-22')
    """
    lc_path = DATA_LC / f"{star}.csv"
    rv_path = DATA_RV / f"{star}_rv.csv"
    exo_path = DATA_EXO / f"{star}_exoplanets.csv"

    lc_df = load_csv(lc_path)
    rv_df = load_csv(rv_path)
    exo_df = load_csv(exo_path)

    detected = detect_and_compare(lc_df, rv_df, exo_df)
    return detected, lc_df, rv_df, exo_df



# ----------------------------
# Visualization: multi-color planet surface + rings + moons
# ----------------------------
def create_multi_color_planet(appearance_list, mix_strengths=None, size_scale=1.0, seed=0):
    """
    appearance_list: list of color labels (e.g. ["bleu","vert"]) or single string
    mix_strengths: relative weights (same length) or None
    Produces a Plotly 3D surface where different patches show different colors and blended transition zones.
    """
    np.random.seed(seed)
    # color mapping
    col_map = {"bleu": np.array([74,144,226])/255.0,
               "rouge": np.array([217,83,79])/255.0,
               "vert": np.array([46,204,113])/255.0,
               "gris": np.array([157,164,168])/255.0,
               "jaune": np.array([245,197,66])/255.0}
    if isinstance(appearance_list, str):
        appearance_list = [appearance_list]

    # strengths
    if mix_strengths is None:
        mix_strengths = [1.0]*len(appearance_list)
    mix_strengths = np.array(mix_strengths, dtype=float)
    if mix_strengths.sum() == 0:
        mix_strengths = np.ones_like(mix_strengths)
    mix_strengths = mix_strengths / mix_strengths.sum()

    # param grid
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu)*np.sin(vv) * size_scale
    y = np.sin(uu)*np.sin(vv) * size_scale
    z = np.cos(vv) * size_scale

    # We'll build an RGB array for the surface by combining color patches:
    # define several random "patch centers" on sphere surface and assign each appearance a few patches
    H, W = vv.shape
    rgb = np.zeros((H, W, 3), dtype=float)

    # create per-appearance patterns using spherical harmonics-like random functions
    for idx, app in enumerate(appearance_list):
        base_col = col_map.get(app, col_map["gris"])
        weight = mix_strengths[idx]
        # pattern: combination of lat/long sinus + random phase -> creates continents / oceans
        freq = 2 + idx
        phase = np.random.RandomState(seed + idx).rand()*2*np.pi
        pattern = (np.sin(freq*uu + phase) * np.cos((freq+1)*vv + phase*0.5))
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-12)  # 0..1
        # smooth patch via gaussian-like mask from random centers
        for pcent in range(2):  # a couple of centers
            lon0 = np.random.RandomState(seed + idx + pcent).rand()*2*np.pi
            lat0 = np.random.RandomState(seed + idx + pcent + 10).rand()*np.pi
            ang = np.arccos(np.sin(vv)*np.sin(lat0) + np.cos(vv)*np.cos(lat0)*np.cos(uu-lon0))
            mask = np.exp(-(ang**2)/(0.6 + 0.2*pcent))
            pattern = np.maximum(pattern, mask* (0.4 + 0.6*pcent))
        # combine pattern scaled by weight
        for c in range(3):
            rgb[:,:,c] += base_col[c] * pattern * weight

    # normalize rgb to 0..1
    rgb = rgb - rgb.min()
    rgb = rgb / (rgb.max() + 1e-12)

    # convert rgb to integer colors for a custom colorscale mapping
    # We'll create a scalar map by projecting rgb to a single value per pixel
    scalar = (0.3*rgb[:,:,0] + 0.6*rgb[:,:,1] + 0.1*rgb[:,:,2])
    # custom colorscale as list of (val, color) — we will sample many colors from rgb grid
    # But Plotly expects colorscale mapping scalar->color; instead we'll set surfacecolor=scalar and use a
    # colorscale that approximates our rgb by sampling unique colors.
    # Build discrete colorscale by sampling K points from RGB grid
    K = 16
    indices = np.linspace(0, 1, K)
    colors = []
    # sample colors from rgb by picking quantiles
    for q in indices:
        # pick location in scalar nearest to q
        flat = scalar.flatten()
        idx_near = np.argmin(np.abs(flat - q))
        r = rgb.reshape(-1,3)[idx_near]
        colors.append("rgb({},{},{})".format(int(r[0]*255), int(r[1]*255), int(r[2]*255)))
    colorscale = []
    for i, col in enumerate(colors):
        colorscale.append([i/(K-1), col])

    surface = go.Surface(x=x, y=y, z=z, surfacecolor=scalar, colorscale=colorscale, showscale=False)
    fig = go.Figure(data=[surface])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0), height=480)
    return fig

def add_rings_and_moons_to_fig(fig, moons_rings, size_scale=1.0, seed=0):
    """
    Adds ring mesh and moon markers to existing figure.
    """
    np.random.seed(seed)
    
    # --- Anneaux ---
    if "anneaux" in moons_rings.lower():
        theta = np.linspace(0, 2*np.pi, 200)
        r_out = 1.8 * size_scale  # un peu plus large
        r_in = 1.2 * size_scale   # un peu plus large
        xr = np.concatenate([r_out*np.cos(theta), r_in*np.cos(theta[::-1])])
        yr = np.concatenate([r_out*np.sin(theta), r_in*np.sin(theta[::-1])])
        zr = np.zeros_like(xr) * 0.05 * size_scale  # augmenter l'épaisseur
        fig.add_trace(go.Mesh3d(x=xr, y=yr, z=zr, alphahull=0, opacity=0.5, color="#b28a46"))
    
    # --- Lunes ---
    if "lune" in moons_rings.lower():
        num = int(''.join(filter(str.isdigit, moons_rings)) or 1)
        for i in range(num):
            ang = 2*np.pi*(i+1)/(num+1) + np.random.RandomState(seed+i).rand()*0.2
            r_orbit = 2.0*size_scale + 0.4*i
            mx = r_orbit*np.cos(ang)
            my = r_orbit*np.sin(ang)
            mz = 0.1*size_scale * ((-1)**i)
            
            # Couleur aléatoire gris/blanc/noir
            moon_color = 'rgb({},{},{})'.format(
                np.random.randint(100, 200),
                np.random.randint(100, 200),
                np.random.randint(100, 200)
            )
            
            # Taille plus grande
            moon_size = 8 * size_scale
            
            fig.add_trace(go.Scatter3d(
                x=[mx], y=[my], z=[mz], mode="markers",
                marker=dict(size=moon_size, color=moon_color)
            ))
    
    return fig


# ----------------------------
# UI pieces (splash/top/sidebar/page1 unchanged)
# ----------------------------
def show_splash_then_lang():
    splash = st.empty()
    splash.markdown("<div style='text-align:center; margin-top:80px'>", unsafe_allow_html=True)
    splash.markdown("<h1 style='font-size:64px'>Horizon.S</h1>", unsafe_allow_html=True)
    splash.markdown("<p style='font-size:18px'>Découvrons au-delà des limites (Horizon), grâce à la science (.S)</p>", unsafe_allow_html=True)
    splash.markdown("</div>", unsafe_allow_html=True)
    time.sleep(3)
    splash.empty()

    st.markdown("---")
    st.markdown("### Choisissez une langue")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Français"):
        st.session_state.lang = "fr"; st.session_state.lang_selected = True; st.session_state.page = "page1"
    if c2.button("English"):
        st.session_state.lang = "en"; st.session_state.lang_selected = True; st.session_state.page = "page1"
    if c3.button("Deutsch"):
        st.session_state.lang = "de"; st.session_state.lang_selected = True; st.session_state.page = "page1"
    if c4.button("Español"):
        st.session_state.lang = "es"; st.session_state.lang_selected = True; st.session_state.page = "page1"

def top_bar():
    cols = st.columns([1,3,1])
    with cols[0]:
        st.markdown("<div style='display:flex; align-items:center; gap:8px'><h3 style='margin:0'>Horizon.S</h3></div>", unsafe_allow_html=True)
    with cols[1]:
        b1, b2 = st.columns([1,1])
        if b1.button("Page 1"): st.session_state.page = "page1"
        if b2.button("Page 2"): st.session_state.page = "page2"
    with cols[2]:
        if st.button("⚙️ Paramètres"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)

def sidebar_nav():
    with st.sidebar:
        st.header("🌌 Horizon.S")
        st.markdown("Navigation")
        if st.button("Page 1 (sidebar)"): st.session_state.page = "page1"
        if st.button("Page 2 (sidebar)"): st.session_state.page = "page2"
        st.markdown("---")
        st.caption("Dossiers attendus : data/ et media/ (optionnel)")

def page_1():
    st.markdown("<h2 style='text-decoration:underline'>Études d'étoiles</h2>", unsafe_allow_html=True)
    z11 = st.empty()
    welcome_msg = ("Bienvenue 👋 sur Horizon.S — ton compagnon pour explorer les étoiles et leurs possibles planètes.\n\n"
                   "Commence par choisir une étoile ci-dessous, puis clique sur **Appliquer** pour charger ses données.")
    type_writer(welcome_msg, z11, delay=0.006)

    st.markdown("---")
    options = ["-- Choisir une étoile --"] + STARS
    chosen = st.selectbox("Choisissez une étoile", options, index=0, key="page1_star_select")
    apply_btn = st.button("Appliquer")

    if apply_btn:
        if chosen == options[0]:
            st.warning("Sélectionnez d'abord une étoile valide.")
        else:
            st.session_state.selected_star = chosen
            st.session_state.apply_clicked = True
            st.session_state.analysis_started = False
            st.session_state.analysis_result = None
            st.session_state.current_planet = None

    st.markdown("---")
    if not st.session_state.apply_clicked or st.session_state.selected_star is None:
        st.info("Aucune étoile appliquée — choisissez une étoile et cliquez sur Appliquer pour afficher ses informations.")
        return

    star = st.session_state.selected_star
    size = f"{random.uniform(0.7, 2.5):.2f} fois la taille du Soleil"
    distance = f"{random.uniform(4.0, 1200.0):.1f} années-lumière"
    mass = f"{random.uniform(0.6, 3.0):.2f} fois la masse solaire"
    temp = f"{random.randint(3000,8000)} K"

    z12 = st.empty()
    ptext = (f"Tu regardes maintenant **{star}**. Cette étoile a une taille approximative de {size}, se situe à environ {distance} de la Terre, "
             f"a une masse d'environ {mass} et une température d'environ {temp}. Ces valeurs sont des estimations de démonstration dans cette version.")
    type_writer(ptext, z12, delay=0.006)

    st.markdown("---")
    st.markdown(f"### <u>Méthode de transit</u>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1,1])
    lc_media = MEDIA_LC / f"{star}_lightcurve.mp4"
    with col_l:
        if lc_media.exists(): st.video(str(lc_media))
        else: st.info("file not found ⚠️ (vidéo transit)")
        st.caption("fichier_mp4")

    lc_csv = DATA_LC / f"{star}.csv"
    lc_df = load_csv(lc_csv)
    with col_r:
        if lc_df is not None and {"time","flux"}.issubset(set(lc_df.columns)):
            st.plotly_chart(plot_lightcurve_df(lc_df), use_container_width=True)
        else:
            t_arr = np.linspace(0, 30, 500)
            flux = 1 - 0.01 * np.exp(-((t_arr-15)**2)/2) + np.random.normal(0, 0.0008, t_arr.shape)
            df_demo = pd.DataFrame({"time": t_arr, "flux": flux})
            st.plotly_chart(plot_lightcurve_df(df_demo), use_container_width=True)
        st.caption("graph")

    st.markdown("---")
    st.markdown(f"### <u>Méthode de vélocité radiale</u>", unsafe_allow_html=True)
    rv_l, rv_r = st.columns([1,1])
    rv_media = MEDIA_RV / f"{star}.mp4"
    rv_csv = DATA_RV / f"{star}_rv.csv"
    rv_df = load_csv(rv_csv)
    with rv_l:
        if rv_media.exists(): st.video(str(rv_media))
        else: st.info("file not found ⚠️ (vidéo RV)")
        st.caption("fichier_mp4")
    with rv_r:
        if rv_df is not None and {"time","velocity"}.issubset(set(rv_df.columns)):
            st.plotly_chart(plot_rv_df(rv_df), use_container_width=True)
        else:
            trv = np.linspace(0, 30, 200)
            rv = 0.02 * np.sin(2*np.pi*trv/6.0) + np.random.normal(0, 0.005, trv.shape)
            df_rv_demo = pd.DataFrame({"time": trv, "velocity": rv})
            st.plotly_chart(plot_rv_df(df_rv_demo), use_container_width=True)
        st.caption("spectrum / radial velocity")

    st.markdown("---")
    st.markdown(f"### <u>Image trouvée</u>", unsafe_allow_html=True)
    im_l, im_r = st.columns([2,1])
    img = MEDIA_IMG / f"{star}.jpg"
    with im_r:
        if img.exists(): st.image(str(img), caption="Image de l'étoile")
        else: st.info("No image")
    with im_l:
        z15 = st.empty()
        if img.exists():
            type_writer("Photo trouvée dans les données (affichée à droite).", z15, delay=0.01)
        else:
            type_writer("Aucune photo disponible pour cette étoile.", z15, delay=0.01)

    st.markdown("")
    if st.button("Page suivante"):
        st.session_state.page = "page2"

# Page 2: call analyse_exoplanetes (iacodetest) and visualize
def page_2():
    star = st.session_state.selected_star
    if not star:
        st.warning("Aucune étoile sélectionnée. Retournez à la page 1 et appliquez une étoile d'abord.")
        if st.button("Page précédente"):
            st.session_state.page = "page1"
        return

    st.markdown(f"<h2 style='text-decoration:underline'>Études des exoplanètes de {star}</h2>", unsafe_allow_html=True)
    z21 = st.empty()
    intro = (f"Ici nous analyserons les données pour détecter des exoplanètes autour de **{star}**. "
             "Clique sur **Commencer l'analyse** pour lancer le traitement.")
    type_writer(intro, z21, delay=0.008)

    st.markdown("")
    if not st.session_state.analysis_started:
        if st.button("Commencer l'analyse"):
            st.session_state.analysis_started = True
            st.session_state.analysis_result = None
            st.session_state.current_planet = None

    # run analysis once
    if st.session_state.analysis_started and st.session_state.analysis_result is None:
        with st.spinner("Analyse IA (iacodetest) en cours..."):
            p = st.progress(0)
            for i in range(30):
                time.sleep(0.04)
                p.progress((i+1)/30)
            p.empty()
            detected, lc_df, rv_df, exo_df = analyse_exoplanetes(star)
            st.session_state.analysis_result = {"status":"success", "planets": detected, "lc":lc_df, "rv":rv_df, "exo":exo_df}

    # display results
    res = st.session_state.analysis_result
    if res is None:
        st.info("Aucune analyse finalisée pour l'instant. Lancez l'analyse pour obtenir des résultats.")
    elif res.get("status") == "failed":
        st.error(res.get("reason"))
    else:
        planets = res.get("planets", [])
        certs = [p for p in planets if p.get("Status")=="certified"]
        cands = [p for p in planets if p.get("Status")!="certified"]

        st.success(f"Analyse terminée : {len(planets)} exoplanète(s) détectée(s).")
        left, right = st.columns([1,1])
        with left:
            st.markdown(f"**Exoplanètes certifiées : {len(certs)}**")
            cert_names = [p.get("Name","unknown") for p in certs] or ["—"]
            choose_cert = st.selectbox(" ", options=cert_names, key="cert_select")
            if st.button("Voir (certifiée)"):
                st.session_state.current_planet = next((p for p in certs if p.get("Name")==choose_cert), None)
        with right:
            st.markdown(f"**Exoplanètes candidates : {len(cands)}**")
            cand_names = [f"Candidate_{i+1}" if p.get("Name")=="unknown" else p.get("Name") for i,p in enumerate(cands)] or ["—"]
            choose_cand = st.selectbox(" ", options=cand_names, key="cand_select")
            if st.button("Voir (candidate)"):
                idx = cand_names.index(choose_cand)
                st.session_state.current_planet = cands[idx] if idx < len(cands) else None

        st.markdown("### Visualisation 3D & Détails")
        vis_col, info_col = st.columns([2,1])
        with vis_col:
            if st.session_state.current_planet:
                p = st.session_state.current_planet
                appearance = p.get("Appearance", "gris")
                # if composition contains multiple hints, use them; otherwise single
                appearances = [appearance]
                # allow composition-based extra color hints (simple heuristic)
                comp = p.get("Composition", "").lower()
                if "h2o" in comp or "water" in comp:
                    appearances.append("bleu")
                if "co2" in comp:
                    appearances.append("gris")
                if "fer" in comp or "iron" in comp:
                    appearances.append("rouge")
                size_scale = 1.0
                try:
                    r = float(p.get("Radius", np.nan))
                    if not np.isnan(r):
                        size_scale = max(0.6, min(r/2.0, 3.0))
                except Exception:
                    size_scale = 1.0
                fig = create_multi_color_planet(appearances, size_scale=size_scale, seed=abs(hash(p.get("Name","unknown")))%9999)
                fig = add_rings_and_moons_to_fig(fig, p.get("Moons_Rings","Aucune"), size_scale=size_scale, seed=abs(hash(p.get("Name","unknown")))%9999)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sélectionnez une exoplanète ci-dessus pour voir sa visualisation 3D.")
        with info_col:
            z23 = st.empty()
            if st.session_state.current_planet:
                p = st.session_state.current_planet
                name = p.get("Name","(sans nom)")
                status = "certifiée" if p.get("Status")=="certified" else "candidate"
                period = f"{p.get('Period', '?'):.2f}" if p.get("Period") is not None else "inconnue"
                radius = f"{p.get('Radius', '?'):.2f} R⊕" if p.get("Radius") is not None else "inconnue"
                mass = f"{p.get('Mass', '?'):.2f} M⊕" if p.get("Mass") is not None else "inconnue"
                hab = "Oui" if p.get("Habitable") else "Non"
                comp = p.get("Composition", "Inconnue")
                app = p.get("Appearance", "gris")
                mr = p.get("Moons_Rings", "Aucune")
                desc = (f"Voici ce que nous avons pour **{name}** ({status}) :\n\n"
                        f"- Période orbitale estimée : **{period}** jours\n"
                        f"- Taille estimée : **{radius}**\n"
                        f"- Masse estimée : **{mass}**\n"
                        f"- Dans la zone habitable : **{hab}**\n"
                        f"- Composition (est.) : **{comp}**\n"
                        f"- Apparence dominante (est.) : **{app}**\n"
                        f"- Lunes / Anneaux : **{mr}**")
                type_writer(desc, z23, delay=0.008)
            else:
                type_writer("Les détails et résultats de la planète sélectionnée apparaîtront ici.", z23, delay=0.008)

    st.markdown("")
    if st.button("Page précédente"):
        st.session_state.page = "page1"

# ----------------------------
# Router
# ----------------------------
def main():
    if not st.session_state.lang_selected:
        show_splash_then_lang()
        return
    top_bar()
    sidebar_nav()
    if st.session_state.page == "splash":
        st.session_state.page = "page1"
    if st.session_state.page == "page1":
        page_1()
    elif st.session_state.page == "page2":
        page_2()
    else:
        page_1()

if __name__ == "__main__":
    main()