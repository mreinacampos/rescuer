# rescuer
RESCUER: Cosmological K-corrections for star clusters
Authors: Marta Reina-Campos and William E.Harris
---

Interactive webtool powered by marimo to calculate the cosmological K-corrections for star clusters: [https://mreinacampos.github.io/rescuer/](https://mreinacampos.github.io/rescuer/)

Since star clusters are well described by single-age and metallicity simple stellar populations, we use the spectral energy distributions from the E-MILES stellar library assuming the BaSTi stellar isochrone models and a Chabrier (2003) IMF. All details about the formalism are decribed in [Reina-Campos & Harris (subm)](https://arxiv.org/abs/2310.02307)

**UPDATE relative to the Streamlit version**: the webapp now allows for multiple K-corrections to be calculated at one go. To use this functionality, the user has to upload a table with the isochrone model, the age in Gyr and the metallicity of the source, the redshift at which it is observed, as well as the observed and rest-frame filter. An example of a table can be found in this repository at example_table.csv .

This app can be also be run locally after installing marimo in a Python environment:
pip install marimo
python script_marimo.py