# rescuer
RESCUER: Cosmological K-corrections for star clusters
Authors: Marta Reina-Campos and William E.Harris
---

Interactive webtool powered by Streamlit to calculate the cosmological K-corrections for star clusters

Since star clusters are well described by single-age and metallicity simple stellar populations, we use the spectral energy distributions from the E-MILES stellar library assuming the BaSTi stellar isochrone models and a Chabrier (2003) IMF. All details about the formalism are decribed in [Reina-Campos & Harris (subm)](https://arxiv.org/abs/2310.02307)

This app can be run locally after installing streamlit in a Python environment:
pip install streamlit
streamlit run streamlit_app.py