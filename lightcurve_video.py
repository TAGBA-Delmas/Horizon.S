import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageEnhance
import os

# --- CONFIGURATION ---
DATA_FILE = "Wolf_359.csv"
STAR_IMAGE = "star_light_model.jpg"
OUTPUT_VIDEO = "Wolf_359_lightcurve.mp4"
DURATION = 40  # secondes
FPS = 30  # images/seconde

# --- CHARGER LES DONNÉES ---
data = pd.read_csv(DATA_FILE)
time = np.array(data['time'])
flux = np.array(data['flux'])

# Normalisation du flux (0 → 1)
flux = (flux - flux.min()) / (flux.max() - flux.min())

# --- CHARGER L’IMAGE DE L’ÉTOILE ---
star_img = Image.open(STAR_IMAGE).convert("RGB")

# --- CRÉER LES IMAGES D’ANIMATION ---
frames = []
n_frames = DURATION * FPS
indices = np.linspace(0, len(time) - 1, n_frames).astype(int)

for i, idx in enumerate(indices):
    # Ajuster luminosité selon le flux
    enhancer = ImageEnhance.Brightness(star_img)
    bright_img = enhancer.enhance(0.5 + flux[idx] * 1.5)  # Variation de 50% à 200%

    # Créer la figure (étoile + graphique)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(bright_img)
    ax[0].axis("off")
    ax[0].set_title("Étoile : intensité lumineuse")

    ax[1].plot(time[:idx], flux[:idx], color="orange")
    ax[1].set_xlim(time.min(), time.max())
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Temps (jours)")
    ax[1].set_ylabel("Flux normalisé")
    ax[1].set_title("Courbe de lumière (Light Curve)")
    ax[1].grid(True, linestyle="--", alpha=0.4)

    # Sauvegarder le frame
    frame_path = f"frame_{i:04d}.jpg"
    plt.tight_layout()
    plt.savefig(frame_path, dpi=100)
    plt.close(fig)
    frames.append(frame_path)

# --- ASSEMBLER EN VIDÉO ---
clip = ImageSequenceClip(frames, fps=FPS)
clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio=False)

# --- SUPPRIMER LES IMAGES TEMPORAIRES ---
for f in frames:
    os.remove(f)

print(f"✅ Vidéo générée : {OUTPUT_VIDEO}")
