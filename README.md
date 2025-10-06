# Horizon.S

**Horizon.S** is an educational and scientific web application designed to introduce students and enthusiasts to modern exoplanet research.
It combines real astronomical data, interactive visualization, and artificial intelligence to make the study of exoplanets clear, accessible, and engaging.

---

## 🌌 Project Overview

Horizon.S allows users to:

* Explore a catalog of stars with real scientific parameters (luminosity, mass, surface temperature, distance, etc.).
* Apply **two main methods of exoplanet detection**:

  1. **Transit Method** – observing the variation of a star’s brightness when a planet passes in front of it.
  2. **Radial Velocity Method** – analyzing the Doppler shift caused by stellar oscillations due to orbiting planets.
* Visualize results through **graphs, videos, and simulations**.
* Understand what these methods reveal about exoplanets (size, orbital period, mass, etc.).

---

## 🚀 Key Features

* **Scientific accuracy** – uses real datasets (NASA, ESA missions such as Kepler, TESS, Gaia).
* **Interactive interface** – implemented with Streamlit for real-time graphs and simulations.
* **AI integration** – automates analysis and identifies potential exoplanets.
* **Educational focus** – provides detailed explanations to help learners understand the science behind the methods.

---

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/TAGBA-Delmas/Horizon.S.git
cd Horizon.S
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application locally:

```bash
streamlit run app.py
```

Then open the link displayed in your terminal (usually `http://localhost:8501/`) in your browser.

---

## 📂 Project Structure

```
Horizon.S/
│-- app.py                  # Main Streamlit application
│-- lightcurve_video.py      # Transit method video generator
│-- rv_video_generator.py    # Radial velocity method video generator
│-- data/                    # Data storage (light curves, RV files, etc.)
│-- media/                   # Media files (images, videos)
│-- ouverture.jpg            # Splash screen image
```

---

## 🎯 Educational Goals

Horizon.S aims to:

* Educate students on how exoplanets are detected in modern astronomy.
* Provide hands-on experience with real astronomical data.
* Inspire curiosity and foster scientific thinking.
* Show how **data science, AI, and physics** can work together to explore the universe.

---

## 📖 License

This project is open-source under the MIT License.
You are free to use, modify, and distribute it for educational and scientific purposes.

---

## ✨ Acknowledgments

* NASA Exoplanet Archive
* ESA missions (Kepler, TESS, Gaia)
* Streamlit community for interactive web applications

---
