import numpy as np
import pandas as pd
import plotly.graph_objects as go
import moviepy.config as mpy_conf
from moviepy.editor import ImageSequenceClip, TextClip, CompositeVideoClip
mpy_conf.IMAGEMAGICK_BINARY = r"C:\Users\tagba\Downloads\ImageMagick-7.1.2-3-Q16-x64-dll\magick.exe"
# --- Paramètres ---
star_name = "Kepler-22"
rv_file = f"data/radial_velocity/{star_name}_rv.csv"
df = pd.read_csv(rv_file)
star = df.iloc[0]

# Paramètres étoiles et planète
R_star = star["st_rad"] / 1000 if not np.isnan(star["st_rad"]) else 1
R_planet = R_star / 10
incl = np.radians(star["pl_orbincl"] if not np.isnan(star["pl_orbincl"]) else 90)
a_planet = star["pl_orbsmax"] if not np.isnan(star["pl_orbsmax"]) else 1
rv_amp = star["pl_rvamp"] if not np.isnan(star["pl_rvamp"]) else 0.05

# Infos à afficher (texte dynamique)
infos = [
    f"Taille étoile: {star['st_rad']:.2f} km",
    f"Masse étoile: {star['st_mass']:.2f} M☉",
    f"Température: {star['st_teff']:.0f} K",
    f"Inclinaison orbite: {star['pl_orbincl']:.1f}°",
    f"Diamètre orbite planète: {star['pl_orbsmax']:.2f} UA"
]

# --- Générer positions ---
frames = []
num_frames = 200
t = np.linspace(0, 2*np.pi, num_frames)

for i in range(num_frames):
    # --- Étoile ---
    theta_star = t[i]
    x_star = 0.05*np.cos(theta_star)
    y_star = 0.05*np.sin(theta_star)*np.cos(incl)
    z_star = 0.05*np.sin(theta_star)*np.sin(incl)
    
    # --- Planète ---
    theta_planet = t[i]*0.8
    x_p = a_planet*np.cos(theta_planet)
    y_p = a_planet*np.sin(theta_planet)*np.cos(incl)
    z_p = a_planet*np.sin(theta_planet)*np.sin(incl)
    
    fig = go.Figure()

    # Orbites
    fig.add_trace(go.Scatter3d(x=[0,x_p*1.1], y=[0,y_p*1.1], z=[0,z_p*1.1],
                               mode="lines", line=dict(color='white')))

    # Étoile (dégradé simple via couleur constante)
    fig.add_trace(go.Scatter3d(x=[x_star], y=[y_star], z=[z_star],
                               mode="markers", marker=dict(size=R_star*20,
                                                           color="orange",
                                                           opacity=0.9)))

    # Planète (ombre simulée par position)
    shadow = 0.5 + 0.5*(z_p/a_planet)
    planet_color = f"rgba({int(200*shadow)}, {int(200*shadow)}, {int(200*shadow)}, 1)"
    fig.add_trace(go.Scatter3d(x=[x_p], y=[y_p], z=[z_p],
                               mode="markers", marker=dict(size=R_planet*20,
                                                           color=planet_color)))
    
    fig.update_layout(scene=dict(bgcolor='black'),
                      margin=dict(l=0,r=0,t=0,b=0),
                      showlegend=False)
    
    filename = f"tmp/frame_{i:03d}.png"
    fig.write_image(filename)
    frames.append(filename)

# --- Bande Doppler RV et texte dynamique via MoviePy ---
clips = []
fps = 20
frame_duration = 1/fps
video_frames = ImageSequenceClip(frames, fps=fps)

# Texte dynamique: afficher chaque info 4s
text_clips = []
start_time = 0
for info in infos:
    txt_clip = TextClip(info, fontsize=24, color='white', font='Arial-Bold')
    txt_clip = txt_clip.set_position(('left','bottom')).set_start(start_time).set_duration(4)
    text_clips.append(txt_clip)
    start_time += 4

# Bande RV
def doppler_color(t):
    # simple oscillation selon RV amplitude
    val = rv_amp*np.sin(2*np.pi*t/4)
    if val>0: return 'red'
    else: return 'blue'

# créer bande RV (rectangle coloré)
from moviepy.video.VideoClip import ColorClip
doppler_clips = []
for i, frame in enumerate(np.linspace(0, 40, int(40*fps))):
    color = doppler_color(frame)
    dop_clip = ColorClip(size=(video_frames.w, 30), color=color).set_start(frame).set_duration(1/fps)
    doppler_clips.append(dop_clip)

# Combiner tout
final = CompositeVideoClip([video_frames, *text_clips, *doppler_clips])
final.write_videofile(f"{star_name}_rv_video.mp4", fps=fps)
