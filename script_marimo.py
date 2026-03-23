# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "astropy==7.2.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(title="RESCUER: K-corrections for SSPs", width="full")

@app.cell
def _(mo):
    mo.md("""
    # RESCUER: Cosmological K-corrections for star clusters
    Written by Marta Reina-Campos and William E. Harris. Based on: [Reina-Campos, M., & Harris, W. E. 2024, MNRAS, 531, 4099](https://scixplorer.org/abs/2024MNRAS.531.4099R/abstract)

    Contact us [here](mailto:marta.reina@usc.es) or through the RESCUE website: https://mreinacampos.github.io/starclusters-in-jwst/
    """)
    return


@app.cell
def _():
    ### Imports and functions

    import marimo as mo
    import os, glob, io, numpy, pandas, matplotlib, sys
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy import units as u
    from astropy import constants
    from astropy.cosmology import Planck18, z_at_value
    import astropy.cosmology.units as cu
    from scipy.interpolate import interp1d

    from urllib.parse import urljoin
    import requests

    if "pyodide" in sys.modules: # WebAssembly -- locally
        base_url = mo.notebook_location()
        
        if "github.io" in str(base_url):  # Only when deployed on GitHub Pages
            # Fetch the raw Python file from GitHub
            raw_url = "https://raw.githubusercontent.com/mreinacampos/rescuer/main/docs/public/functions_Kcorrection.py"
            code = requests.get(raw_url).text
        else:  # Local testing, use local file
            code = requests.get(urljoin(str(base_url), "public/functions_Kcorrection.py")).text
        
        exec(code, globals())
    else: # locally in VS Code
        sys.path.append(os.path.join(".", "public"))
        from functions_Kcorrection import func_kcorrection_lambda

    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['font.size'] = 18.0
    matplotlib.rcParams['legend.fontsize'] = 16.0
    return (
        Planck18,
        base_url,
        constants,
        cu,
        fits,
        func_kcorrection_lambda,
        glob,
        interp1d,
        io,
        matplotlib,
        mo,
        numpy,
        os,
        pandas,
        plt,
        requests,
        sys,
        u,
        urljoin,
        z_at_value,
    )


@app.cell
def _(
    Planck18,
    base_url,
    constants,
    fits,
    glob,
    mo,
    numpy,
    os,
    requests,
    sys,
    u,
    urljoin,
    z_at_value,
):
    def select_emiles_model(age, mh, name_slope, name_label, inpath):
        # format the name
        if mh < 0:
            sign_mh = "m"
        else:
            sign_mh = "p"
        name_mh = "Z{:s}{:.2f}".format(sign_mh, numpy.abs(mh))
        name_age = "T{:07.4f}".format(age)

        fname_table = os.path.join(inpath, '{:s}{:s}{:s}_{:s}0.00_baseFe.fits'.format(name_slope,
                                                                                      name_mh, 
                                                                                      name_age, 
                                                                                      name_label))
        try:
            if "pyodide" in sys.modules:
                response = requests.get(fname_table)
                hdul = fits.open(io.BytesIO(response.content))
            else:
                hdul = fits.open(fname_table)
            sed = {}
            # wavelength range for the E-MILES models - Å
            sed["lambda"] = numpy.linspace(1680.2, 49999.4, len(hdul[0].data)) * u.angstrom
            # transform to frequencies
            sed["nu"] = sed["lambda"].to(u.Hz, equivalencies=u.spectral())
            ### LUMINOSITY
            # SSP spectra is output in Lλ/ LSun MSun^-1 Å^-1 units
            sed["lum_angstrom"] = hdul[0].data * u.solLum / u.angstrom / u.solMass
            # convert to erg/s/MSun/A
            sed["lum_angstrom"] = sed["lum_angstrom"].to(u.erg / u.s / u.angstrom / u.solMass)

            # transform to Hz^-1 -> f_nu = f_lambda*lambda/(nu = c/lambda)
            convert_to_hz = numpy.power(sed["lambda"], 2) / constants.c.to("angstrom s^-1")
            sed["lum_hz"] = sed["lum_angstrom"] * convert_to_hz / u.s / u.Hz
            hdul.close()
            return sed
        except: 
            mo.md("{:s} could NOT be found".format(fname_table))
            return None


    # define the dictionary holding the AB standard source information
    def define_standard_source(sed):
        # define the standard source
        stdsource = {}
        stdsource["nu"] = sed["nu"].copy() # Hz
        stdsource["lambda"] = sed["lambda"].copy() # A
        # spectral density of flux for the zero-magnitude or “standard” source
        # For AB magnitudes (Oke & Gunn 1983), is a hypothetical constant source with gAB(ν) = 3631 Jy (where 1 Jy = 10−26 W m−2 Hz−1 = ν 10−23 erg cm−2 s−1 Hz−1) at all frequencies ν
        stdsource["gnu_AB"] = (3631 * u.Jy).to(u.erg * u.s**-1 * u.cm**-2 * u.Hz**-1) # Jy = 10−23 erg cm−2 s−1 Hz−1
        stdsource["flux_hz"] = numpy.ones(shape=len(stdsource["nu"]))*stdsource["gnu_AB"] # erg/s/cm^/Hz
        convert_to_angstrom = numpy.power(stdsource["lambda"], -2)*constants.c.to("angstrom s^-1") * u.s
        stdsource["flux_angstrom"] = (stdsource["flux_hz"] * u.Hz)*convert_to_angstrom

        return stdsource

    # function to load the files with the filter throughputs
    def load_filter_data(name_filter):
        # load the name of all the files
        ### load the available filters
        if "pyodide" in sys.modules: # WebAssembly
            if "github.io" in str(base_url):  # Only when deployed on GitHub Pages
                # Fetch the raw Python file from GitHub
                filter_list_url = "https://raw.githubusercontent.com/mreinacampos/rescuer/main/docs/public/filter_list.txt"
                _base_url = "https://raw.githubusercontent.com/mreinacampos/rescuer/main/docs/"
            else:  # Local testing, use local file
                filter_list_url = urljoin(str(mo.notebook_location()), "public/filter_list.txt")
                _base_url = mo.notebook_location()
                
            text = requests.get(filter_list_url).text
            filter_files = [line.strip() for line in text.splitlines() if line.strip()]
            ls_files_filters = [urljoin(str(_base_url), f"public/Filters/{name}") for name in filter_files]

        else:
            ls_files_filters = glob.glob(os.path.join(".", "public", "Filters", "*.dat"))
            ls_files_filters.sort()
    
        # output the data in a dictionary
        data_filter = {}
        # keep only the FXYZW
        range_filter = name_filter.split("_")[-1]
        # find the file
        for name in ls_files_filters:
            if range_filter == name.split(".")[-2]:
                if "pyodide" in sys.modules:
                    response = requests.get(name)
                    tmp = numpy.genfromtxt(io.StringIO(response.text), comments="#")
                else:
                    tmp = numpy.genfromtxt(name, comments="#")
                data_filter["lambda"] = tmp[:,0] * u.angstrom # Angstrom
                data_filter["nu"] = data_filter["lambda"].to(u.Hz, equivalencies=u.spectral()) # Hz
                data_filter["curve"] = tmp[:,1] # unitless
                break
        return data_filter  


    def plancks_law(wavelength, T):
        # wavelength in angstrom and T in K
        # spectral radiance (the power per unit solid angle and per unit of area normal to the propagation)
        # density of frequency nu  radiation per unit frequency at thermal equilibrium at temperature T
        # Units: power / [area × solid angle × frequency] -> power = energy/time.

        nu = constants.c.to("angstrom s^-1") / wavelength # 1/s = Hz   
        spectral_radiance = (2.0 * constants.h.to("erg s") * nu*nu*nu)/(constants.c*constants.c*(numpy.exp(constants.h*nu/(constants.k_B*T)) - 1))

        # units: (ergs / s) sr^-1 Hz^-1
        return spectral_radiance.to("erg cm^-2") / u.s / u.Hz / u.sr

    # function to check that the selected age and redshift are compatibles
    def check_redshift_age_compatibility(age, z):
        # at what redshift they would be the oldest SSPs? - assumption: 500Myr post-BigBang for formation
        age += 0.5 # assumption: SSPs take at least 500Myr to start forming from the Big Bang
        z_oldest = z_at_value(Planck18.age, age * u.Gyr) 
        if z > z_oldest: return False
        else: return True

    def define_edges_region(dict_choices, dict_options):
        # define the edges of the parameter space that fall within the uncertainties in age and metallicity

        # lower and upper uncertainties - target values, depend on the model sampling
        errors_sed = numpy.asarray([[[dict_choices["sigma_mh"]],[dict_choices["sigma_age"]*dict_choices["age"]/100]],
                                    [[dict_choices["sigma_mh"]],[dict_choices["sigma_age"]*dict_choices["age"]/100]]]) 

        # define the edges of the region
        mh_edges = [dict_choices["mh"] - errors_sed[:, 0][0][0], dict_choices["mh"]+errors_sed[:, 0][1][0]]
        age_edges = [dict_choices["age"] - errors_sed[:, 1][0][0], dict_choices["age"] + errors_sed[:, 1][1][0]]
        for ind in [0,1]: # check that both mh and age edges exist in the list of models
            # check the mh edges and correct the uncertainties
            if mh_edges[ind] not in dict_options[dict_choices["isochrone_model"]]["ls_models_mh"]:
                mh_edges[ind] = dict_options[dict_choices["isochrone_model"]]["ls_models_mh"][numpy.abs(dict_options[dict_choices["isochrone_model"]]["ls_models_mh"] - mh_edges[ind]).argmin()]
                errors_sed[:, 0][ind] = numpy.abs(mh_edges[ind] - dict_choices["mh"])
                dict_choices["sigma_mh"] = errors_sed[:, 0].flatten().max() # in dex
            # check the age edges and correct the uncertainties
            if age_edges[ind] not in dict_options[dict_choices["isochrone_model"]]["ls_models_ages"]:
                age_edges[ind] = dict_options[dict_choices["isochrone_model"]]["ls_models_ages"][numpy.abs(dict_options[dict_choices["isochrone_model"]]["ls_models_ages"] - age_edges[ind]).argmin()]
                errors_sed[:, 1][ind] = numpy.abs(age_edges[ind] - dict_choices["age"])
                dict_choices["sigma_age"] = errors_sed[:, 1].flatten().max()/dict_choices["age"] # in relative space
        return mh_edges, age_edges

    def select_models_region(dict_choices, dict_options, mh_edges, age_edges):
        # select the SED models that fall within the region of interest
        mask_subsample_models_age = (dict_options[dict_choices["isochrone_model"]]["ls_models_ages"] <= age_edges[1]) \
            & (dict_options[dict_choices["isochrone_model"]]["ls_models_ages"] >= age_edges[0])
        mask_subsample_models_mh = (dict_options[dict_choices["isochrone_model"]]["ls_models_mh"] <= mh_edges[1]) \
            & (dict_options[dict_choices["isochrone_model"]]["ls_models_mh"] >= mh_edges[0])
        subsample_models_age = dict_options[dict_choices["isochrone_model"]]["ls_models_ages"][mask_subsample_models_age]
        subsample_models_mh = dict_options[dict_choices["isochrone_model"]]["ls_models_mh"][mask_subsample_models_mh]
        return subsample_models_age, subsample_models_mh


    return (
        check_redshift_age_compatibility,
        define_edges_region,
        define_standard_source,
        load_filter_data,
        plancks_law,
        select_emiles_model,
        select_models_region,
    )


@app.cell
def _(matplotlib, numpy, plt):
    # function to show the mulinosity of the chosen SSP
    def plot_luminosities(age, mh, emiles_sed, z, obs_filter, rf_filter, target_sed, doing_bb = False):
        matplotlib.use('agg')  # required, use a non-interactive backend

        fig, axs = plt.subplots(1,2, figsize = (18,5.5), sharex = True, sharey = True, squeeze = False)
        axs = axs.flatten()

        left = 0.1; right = 0.87
        top = 0.98; bottom = 0.1
        hspace = 0.; wspace = 0.0

        max_lum = emiles_sed[target_sed]["lum_angstrom"].max()

        ### FIRST PANEL: homochromatic K-corrections 
        for i, ax in enumerate(axs):
            if i == 0:
                # add the intrinsic luminosities
                for _key in emiles_sed.keys():
                    if _key == target_sed: 
                        color = "k"; lw = 1; ls = "-"; alpha = 1; label = "Intrinsic luminosity"
                        # add the redhsifted SED only for the target SED
                        ax.plot(emiles_sed[_key]["lambda"].to("micron")*(1+z), emiles_sed[_key]["lum_angstrom"]/max_lum/(1+z),
                            ls = ls, alpha = 0.3, c = "k", lw = lw, label = "Observed at z={:.1f}".format(z))
                        ax.plot(emiles_sed[_key]["lambda"].to("micron"), emiles_sed[_key]["lum_angstrom"]/max_lum,
                            color = color, lw = lw, ls = ls, alpha = alpha, label = label)
                    #else: 
                    #    color = "k" ; lw = 1; ls = ":"; alpha = 0.1; label = ""

            elif i == 1:
                # approximation to the integrals if the filters are deltas
                yy_em = emiles_sed[target_sed]["lum_angstrom"]*numpy.power(emiles_sed[target_sed]["lambda"], 2)
                yy_obs = emiles_sed[target_sed]["lum_angstrom"]*numpy.power(emiles_sed[target_sed]["lambda"]*(1+z), 2)/(1+z)

                ax.plot(emiles_sed[target_sed]["lambda"].to("micron"), yy_em/yy_obs.max(),
                    color = "k", lw = 1, ls = "-", label = r"$\lambda^2$ L$_{\lambda}(\lambda)$")
                ax.plot(emiles_sed[target_sed]["lambda"].to("micron")*(1+z), yy_obs/yy_obs.max(),
                       ls = "-", alpha = 0.3, c = "k", label = r"$(1+z)^{-1} \lambda^2$ L$_{\lambda}[(1+z)^{-1}\lambda]$")

            # add the wavelength range covered by each filter
            ax.axvspan(obs_filter['lambda'].to('micron').value.min(), obs_filter['lambda'].to('micron').value.max(), color = "C0", alpha = 0.1, label = "Observed filter")
            ax.axvspan(rf_filter['lambda'].to('micron').value.min(), rf_filter['lambda'].to('micron').value.max(), color = "C1", alpha = 0.1, label = "Rest-frame filter")


        # add the K-correction arrow
        obs_lambda = 0.5*(obs_filter['lambda'].to('micron').min() + obs_filter['lambda'].to('micron').max()) # mid-point of the wavelength range
        idx_lambda = numpy.argmin(numpy.abs(emiles_sed[target_sed]["lambda"].to("micron")*(1+z)-obs_lambda.to("micron"))) # find the index of the wavelength closest to the midpoint
        obs_lum = (emiles_sed[target_sed]["lum_angstrom"]/yy_obs.max()/(1+z))[idx_lambda] * numpy.power(obs_lambda.to("angstrom"), 2)
        em_lambda = 0.5*(rf_filter['lambda'].to('micron').min() + rf_filter['lambda'].to('micron').max())# mid-point of the wavelength range
        idx_lambda = numpy.argmin(numpy.abs(emiles_sed[target_sed]["lambda"].to("micron")-em_lambda.to("micron"))) # find the index of the wavelength closest to the midpoint
        em_lum = (emiles_sed[target_sed]["lum_angstrom"]/yy_obs.max())[idx_lambda] * numpy.power(em_lambda.to("angstrom"), 2)
        # annotate the behaviour of the K-correction
        axs[1].annotate("K", xy=(obs_lambda.to("micron").value+0.05, obs_lum.value+0.05), va = "center", ha = "left", color = "C3", zorder = 10)
        axs[1].plot(obs_lambda.to("micron").value, obs_lum, marker = "o", color = "C3", markersize = 8, zorder = 10)
        axs[1].annotate("", xy=(em_lambda.value, em_lum.value), xytext=(obs_lambda.value, obs_lum.value),
                           va = "center", ha = "center", 
                           arrowprops=dict(color='C3', lw = 3, arrowstyle = "->"), zorder = 10)

        # annotate the SSP used
        if doing_bb:
            axs[0].annotate(r"Blackbody curve of $T={:.1f}~\rm K$".format(doing_bb),
                    xy = (0.98, 0.97), xycoords = "axes fraction", va = "top", ha = "right")
        else:
            axs[0].annotate(r"SSP of {:.2f} Gyr and [M/H] = {:.2f}".format(age, mh),
                    xy = (0.98, 0.97), xycoords = "axes fraction", va = "top", ha = "right")

        # format all axes
        for i, ax in enumerate(axs):
            ax.set_xlabel(r"Wavelength [$\mu$m]")
            ax.tick_params(bottom = True, left= True, right = True, top = True, axis = "both", which = "both")
            ax.set_ylim(0, None)
            #ax.set_ylabel(r"$(1+z)^{{-1}}L_{{\lambda}}\left(\frac{{\lambda}}{{(1+z)}}\right)$ [{:.0e} ergs s$^{{-1}}$ $\AA^{{-1}}$ M$_{{\odot}}^{{-1}}$]".format(max_lum.value))     
            if i == 0:
                ax.legend(loc = "center right", frameon = True)
                ax.set_ylabel("Normalised SEDs at a given distance")   
            elif i == 1:
                ax.legend(loc = "upper right", frameon = True)

            ax.set_xlim(0, None)
            ax.minorticks_on()

        # format the entire figure
        fig.subplots_adjust(left = left, top = top, bottom = bottom, right = right, hspace = hspace, wspace = wspace)

        return max_lum, yy_obs.max(), fig

    return (plot_luminosities,)


@app.cell
def _(base_url, glob, mo, numpy, os, requests, sys, urljoin):
    ### PREPARE variables
    # define variables describing the E-MILES models
    #name_isochrone = "BASTI" # "PADOVA00"
    name_imf = "CH" # Chabrier 2003
    name_slope = "Ech1.30" # slope of the IMF

    dict_options = {
        "BASTI": {
            "name_label": "iTp",
            "ls_models_mh": numpy.asarray([-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25, 0.06, 0.15, 0.26, 0.4]),
            "ls_models_ages": numpy.asarray([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
        },
        "PADOVA00": {
            "name_label": "iPp",
            "ls_models_mh": numpy.asarray([-2.32, -1.71, -1.31, -0.71, -0.4, 0.0, 0.22]),
            "ls_models_ages": numpy.asarray([0.0631, 0.0708, 0.0794, 0.0891, 0.1, 0.1122, 0.1259, 0.1413, 0.1585, 0.1778, 0.1995, 0.2239, 0.2512, 0.2818, 0.3162, 0.3548, 0.3981, 0.4467, 0.5012, 0.5623, 0.631, 0.7079, 0.7943, 0.8913, 1.0, 1.122, 1.2589, 1.4125, 1.5849, 1.7783, 1.9953, 2.2387, 2.5119, 2.8184, 3.1623, 3.5481, 3.9811, 4.4668, 5.0119, 5.6234, 6.3096, 7.0795, 7.9433, 8.9125, 10.0, 11.2202, 12.5893])
        }
    }
    ### load the available filters
    if "pyodide" in sys.modules: # WebAssembly
        if "github.io" in str(base_url):  # Only when deployed on GitHub Pages
            # Fetch the raw Python file from GitHub
            filter_list_url = "https://raw.githubusercontent.com/mreinacampos/rescuer/main/docs/public/filter_list.txt"
            _base_url = "https://raw.githubusercontent.com/mreinacampos/rescuer/main/docs/"
        else:  # Local testing, use local file
            filter_list_url = urljoin(str(mo.notebook_location()), "public/filter_list.txt")
            _base_url = mo.notebook_location()
            
        text = requests.get(filter_list_url).text
        filter_files = [line.strip() for line in text.splitlines() if line.strip()]
        _ls_filters = [urljoin(str(_base_url), f"public/{name}") for name in filter_files]

    else:
        _ls_filters = glob.glob(os.path.join(".", "public", "Filters", "*.dat"))

    ls_filters = sorted(["_".join(f.split("/")[-1].split(".")[:-1]) for f in _ls_filters])
    ls_filters.sort()
    return dict_options, ls_filters, name_imf, name_slope


@app.cell
def _(mo):
    ### SIDEBAR
    # place the selectboxes on the sidebar
    dr_isochrone = mo.ui.dropdown(options=["BASTI", "PADOVA00"], value="BASTI", label="Isochrone Model")
    return (dr_isochrone,)


@app.cell
def _(dict_options, dr_isochrone, ls_filters, mo):
    dr_age = mo.ui.dropdown(options=[str(a) for a in dict_options[dr_isochrone.value]["ls_models_ages"]], value="10.0", label="Age [Gyr]")
    dr_mh = mo.ui.dropdown(options=[str(m) for m in dict_options[dr_isochrone.value]["ls_models_mh"]], value="-2.27", label="Metallicity [M/H]")
    dr_redshift = mo.ui.number(value=0.0, start=0.0, stop=1.0, step=0.01, label="Redshift [z]")
    dr_obs_filter = mo.ui.dropdown(options=ls_filters, label="Observed Filter", value = ls_filters[0])
    dr_rf_filter = mo.ui.dropdown(options=ls_filters, label="Rest-frame Filter", value = ls_filters[0])
    dr_add_uncertainties = mo.ui.checkbox(label="Add uncertainties?")
    return (
        dr_add_uncertainties,
        dr_age,
        dr_mh,
        dr_obs_filter,
        dr_redshift,
        dr_rf_filter,
    )


@app.cell
def _(dr_add_uncertainties, mo):
    dr_sigma_age = mo.ui.number(value=20.0, label="Relative age uncertainty [%]") if dr_add_uncertainties.value else "No uncertainties on the age"
    dr_sigma_mh = mo.ui.number(value=0.3, label="Metallicity uncertainty [dex]") if dr_add_uncertainties.value else "No uncertainties on the metallicity"
    return dr_sigma_age, dr_sigma_mh


@app.cell
def _(mo):
    dr_file_button = mo.ui.file(kind="button",filetypes=[".csv"], multiple=False)
    dr_file_area = mo.ui.file(kind="area",filetypes=[".csv"], multiple=False)
    return dr_file_area, dr_file_button


@app.cell
def _(
    dr_add_uncertainties,
    dr_age,
    dr_file_area,
    dr_file_button,
    dr_isochrone,
    dr_mh,
    dr_obs_filter,
    dr_redshift,
    dr_rf_filter,
    dr_sigma_age,
    dr_sigma_mh,
    mo,
):
    sidebar_content = mo.sidebar(
            item=mo.vstack([mo.md("## Configuration"),
                            mo.md("---"),
                            mo.md("### For single K-corrections:"),
                            dr_isochrone, dr_age, dr_mh, dr_redshift,
                            dr_obs_filter, dr_rf_filter, 
                            dr_add_uncertainties, dr_sigma_age, dr_sigma_mh,
                            mo.md("---"),
                            mo.md("### For multiple K-corrections:"), dr_file_button, dr_file_area] ) )
    return (sidebar_content,)


@app.cell
def _(
    dr_add_uncertainties,
    dr_age,
    dr_isochrone,
    dr_mh,
    dr_obs_filter,
    dr_redshift,
    dr_rf_filter,
    dr_sigma_age,
    dr_sigma_mh,
):
    dict_choices = {}
    dict_choices["isochrone_model"] = dr_isochrone.value
    dict_choices["age"] = float(dr_age.value)
    dict_choices["mh"] = float(dr_mh.value)
    dict_choices["redshift"] = float(dr_redshift.value)
    dict_choices["obs_filter"] = dr_obs_filter.value
    dict_choices["rf_filter"] = dr_rf_filter.value
    dict_choices["add_uncertainties"] = dr_add_uncertainties.value
    if dict_choices["add_uncertainties"]:
        dict_choices["sigma_age"] = float(dr_sigma_age.value)
        dict_choices["sigma_mh"] = float(dr_sigma_mh.value)
    return (dict_choices,)


@app.cell
def _(sidebar_content):
    sidebar_content
    return


@app.cell
def _(dr_file_area, dr_file_button, io, pandas, pd):
    def parse_uploaded_csv_with_comments(uploaded):
        # uploaded from dr_file_button / dr_file_area
        if not uploaded:
            return [], None
        content = uploaded[0].contents
        if isinstance(content, bytes):
            text = content.decode('utf-8')
        else:
            text = str(content)

        # find all the lines that are comments at the beginning of the file
        comment_lines = []
        data_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('##'):
                comment_lines.append(stripped.lstrip('#').strip())
            else:
                data_lines.append(line)

        if not data_lines:
            return comment_lines, None
        # convert the uploaded file into a pandas dataframe
        header_line = comment_lines[-1] if comment_lines else None
        if header_line:
            data_text = '\n'.join(data_lines)
            df = pandas.read_csv(io.StringIO(data_text), header=None, names=[c.strip() for c in header_line.split(',')])
        else:
            df = pd.read_csv(io.StringIO('\n'.join(data_lines)))
        return comment_lines, df

    # Example use:
    button_comments, button_df = parse_uploaded_csv_with_comments(dr_file_button.value)
    area_comments, area_df = parse_uploaded_csv_with_comments(dr_file_area.value)

    # Choose whichever data frame is present:
    uploaded_df = button_df if button_df is not None else area_df

    mode = "single" if uploaded_df is None else "multiple"
    return mode, uploaded_df


@app.cell
def _(dict_choices, dr_file_area, dr_file_button, ls_filters, mo):
    # is the app in its original state? Check if the dropdowns are still in their default value
    original = (dict_choices["isochrone_model"] == "BASTI")&(dict_choices["age"] == 10.0)&(dict_choices["mh"] == -2.27)&(dict_choices["redshift"] == 0.0)&(dict_choices["obs_filter"] == ls_filters[0])&(dict_choices["rf_filter"] == ls_filters[0])&(len(dr_file_button.value) == 0)&(len(dr_file_area.value) == 0)
    mo.md('## Usage instructions: \n This webapp has two modes of use:\n \n ### For a **single K-correction**\n\n On the left-hand sidebar, select:\n\n 1. The isochrone model,\n\n 2. the age and metallicity of the SSP,\n\n 4. the redshift, \n\n 4. the observed and the rest-frame filters.\n\n An interactive table and a figure will pop out with the K-correction and an interpretation.\n \n ### For **multiple K-corrections**: \n\n Using the button/area on the left-hand sidebar, upload a table with columns indicating (in order): \n\n * the isochrone model, \n\n * the age and metallicity of the SSP,\n\n * its redshift, \n\n * as well as the observed and the rest-frame filters (an example is provided in the repository under example_table.txt). \n\n An interactive table will pop out with the K-corrections.') if original else None
    return


@app.cell
def _(mo, mode, uploaded_df):
    mo.md("This is the uploaded file:"), mo.ui.table(uploaded_df) if mode == "multiple" else None
    return


@app.cell
def _(Planck18, check_redshift_age_compatibility, cu, dict_choices, mo, u):
    # check that the age is compatible with the age of the Universe
    compatible = check_redshift_age_compatibility(float(dict_choices["age"]), dict_choices["redshift"])
    max_age = Planck18.age(dict_choices["redshift"] * cu.redshift).to(u.Gyr) - 0.5 * u.Gyr
    mo.md("### ⚠️ Age and Redshift are incompatible in Planck18 cosmology:\n **The adopted age is not measurable at z={:.3f} in the Planck18 cosmology. The maximum allowed age at this redshift is {:.2f} (assuming 500 Myr for formation after the Big Bang). Please try again.**".format(dict_choices["redshift"], max_age)) if not compatible else None
    return (compatible,)


@app.cell
def _(
    compatible,
    constants,
    define_edges_region,
    define_standard_source,
    dict_choices,
    dict_options,
    func_kcorrection_lambda,
    interp1d,
    load_filter_data,
    mo,
    name_imf,
    name_slope,
    numpy,
    os,
    pandas,
    plancks_law,
    plot_luminosities,
    select_emiles_model,
    select_models_region,
    sys,
    u,
    urljoin,
):
    if compatible:

        # load the filter data
        obs_filter = load_filter_data(dict_choices["obs_filter"])
        rf_filter = load_filter_data(dict_choices["rf_filter"])

        # load the data
        if "pyodide" in sys.modules: # WebAssembly
            if "github.io" in str(base_url):  # Only when deployed on GitHub Pages
                # Fetch the raw Python file from GitHub
                _url = "https://raw.githubusercontent.com/mreinacampos/rescuer/main/docs/"
            else:  # Local testing, use local file
                _url = str(mo.notebook_location())
            inpath = urljoin(_url, os.path.join("public", "SEDs_E-MILES", "EMILES_{:s}_BASE_{:s}_FITS".format(dict_choices["isochrone_model"], name_imf)))
        else:
            inpath = os.path.join(os.curdir, "public", "SEDs_E-MILES", "EMILES_{:s}_BASE_{:s}_FITS".format(dict_choices["isochrone_model"], name_imf))

        # define the target SED
        #target_sed = numpy.asarray([dict_choices["mh"], dict_choices["age"]])
        emiles_sed = {}
        # load the target SED
        name_mh = "Z{:.2f}".format(dict_choices["mh"])    
        name_age = "T{:07.4f}".format(dict_choices["age"])   
        _key = "{:s}_{:s}".format(name_mh, name_age)            
        emiles_sed[_key] = select_emiles_model(age = dict_choices["age"],
                                               mh = dict_choices["mh"],
                                        name_slope = name_slope, 
                                        name_label = dict_options[dict_choices["isochrone_model"]]["name_label"],
                                        inpath = inpath)
        dict_choices["target_sed"] = _key
        # load the other models if needed 
        if dict_choices["add_uncertainties"]:
            # given the uncertainties, define the region and select the right models
            age_edges, mh_edges = define_edges_region(dict_choices, dict_options)
            subsample_models_age, subsample_models_mh = select_models_region(dict_choices, dict_options, age_edges, mh_edges)
            for age in subsample_models_age:
                name_age = "T{:07.4f}".format(age)   
                for mh in subsample_models_mh:
                    name_mh = "Z{:.2f}".format(mh)    
                    _key = "{:s}_{:s}".format(name_mh, name_age)
                    if _key not in emiles_sed.keys(): # if the model has not been loaded yet
                        emiles_sed[_key] = select_emiles_model(age = age, mh = mh,
                                                    name_slope = name_slope, 
                                                    name_label = dict_options[dict_choices["isochrone_model"]]["name_label"],
                                                    inpath = inpath)
        dict_choices["list_seds"] = list(emiles_sed.keys())

        doing_blackbody = 0
        # if we're using the longest range filters, then overwrite the E-MILES SED and use a blackbody approximation
        if "F444W" in dict_choices["obs_filter"] or "F444W" in dict_choices["rf_filter"]:
            for _key in emiles_sed.keys():
                # interpolate the E-MILES SED to get the peak wavelength
                interpolated = interp1d(emiles_sed[_key]["lambda"], emiles_sed[_key]["lum_angstrom"], axis = 0, fill_value = 'extrapolate')
                down_lambda = numpy.linspace(emiles_sed[_key]["lambda"].min(), emiles_sed[_key]["lambda"].max(), int(len(emiles_sed[_key]["lambda"])/2))
                downsampled = interpolated(down_lambda)
                peak_lambda = down_lambda[ downsampled == downsampled.max()][0]
                # applying Wien's law to get the temperature
                T = ( 2898 * u.micron * u.K ).to(u.AA * u.K) / peak_lambda
                if _key == dict_choices["target_sed"]:
                    doing_blackbody = T.value
                # resample the array containing the wavelengths
                max_wavelength = numpy.max([obs_filter["lambda"].max().value, rf_filter["lambda"].max().value])*rf_filter["lambda"].unit
                emiles_sed[_key]["lambda"] = numpy.linspace(emiles_sed[_key]["lambda"].min(), max_wavelength, len(emiles_sed[_key]["lambda"]))
                # determine a blackbody spectrum - units: erg/s/sr/cm^2/Hz
                blackbody_nu = plancks_law(emiles_sed[_key]["lambda"], T = T)
                # transforming quantities using nu*flux_nu = flux_lambda*lambda and lambda*nu = c
                blackbody_lambda = (blackbody_nu * u.Hz)*numpy.power(emiles_sed[_key]["lambda"], -2)*constants.c.to("angstrom s^-1") * u.s
                # units: erg/s/sr/A
                emiles_sed[_key]["lum_angstrom"] = 4*numpy.pi*numpy.power((10 * u.pc).to("cm"), 2)*blackbody_lambda

            mo.md("**[WARNING]:** One of the selected filters exceeds the wavelength range covered by the E-MILES SEDs. The code is mimicking the chosen SED with a blackbody approximation of $T={:.1f}$ so it peaks at the same wavelength.".format(doing_blackbody))

        # define the standard source - AB magnitudes
        stdsource = define_standard_source(emiles_sed[dict_choices["target_sed"]])
        target_kcorr = 0; ls_kcorr = []
        for _key in emiles_sed.keys():
            # calculate the K-correction
            kcorr = func_kcorrection_lambda(emiles_sed[_key]["lambda"],
                                            emiles_sed[_key]["lum_angstrom"],
                                            stdsource["lambda"],
                                            stdsource["flux_angstrom"], 
                                            obs_filter['lambda'], obs_filter['curve'],
                                            rf_filter['lambda'], rf_filter['curve'], dict_choices["redshift"])
            ls_kcorr.append(kcorr)
            if _key == dict_choices["target_sed"]:
                target_kcorr = kcorr

            if dict_choices["add_uncertainties"]:
                # calculate the uncertainties as the distances to the 10th-90th percentiles
                per10th = numpy.min([numpy.percentile(ls_kcorr, 10), target_kcorr])
                per90th = numpy.max([numpy.percentile(ls_kcorr, 90), target_kcorr])
                dict_choices["sigma_kcorr"] = [per10th, per90th]

        if numpy.isnan(target_kcorr) or numpy.isinf(target_kcorr):
            explanation = "This SED and filter combination is misbehaving; check coverage."
        elif target_kcorr < 0:
            explanation = "A negative K-correction indicates observed flux brighter than rest-frame."
        elif target_kcorr > 0:
            explanation = "A positive K-correction indicates observed flux dimmer than rest-frame."
        else:
            explanation = "Null K-correction: observed flux equals rest-frame."

        dict_table = {
            "Age [Gyr]": dict_choices["age"],
            "[M/H]": dict_choices["mh"],
            "Redshift" : dict_choices["redshift"],
            "Obs filter": dict_choices["obs_filter"], 
            "RF filter": dict_choices["rf_filter"],
            "K-correction [AB mag]": numpy.round(target_kcorr, 5)}
        if dict_choices["add_uncertainties"]:
            dict_table["10th percentile"] = numpy.round(dict_choices["sigma_kcorr"][0], 5)
            dict_table["90th percentile"] = numpy.round(dict_choices["sigma_kcorr"][-1], 5)

        df = pandas.DataFrame(dict_table, index = [0])


        _max_lum, _yy_max, _fig = plot_luminosities(dict_choices["age"],
                                dict_choices["mh"],
                                emiles_sed, 
                                z = dict_choices["redshift"],
                                obs_filter = obs_filter, 
                                rf_filter = rf_filter,
                                target_sed = dict_choices["target_sed"],
                                doing_bb = doing_blackbody)

    mo.vstack([
        mo.md("**[Table]** Summary of selected inputs and K-correction result"),
        mo.ui.table(df),
        mo.md(f"**[Interpretation]** {explanation}"),
        mo.md("**[Image]:** Expected behaviour of the K-correction on the SED of a SSP of {:.2f} Gyr and [M/H] = {:.2f} observed at z = {:.2f}: (top) Intrinsic luminosity of the SSP from E-MILES (black) and its redshifted curve (grey) as a function of wavelength, and (bottom) behaviour of the K-correction. The vertical shaded areas indicate the wavelength range covered by each filter. The correction given by the K value is indicated by the red arrow.".format(dict_choices["age"], dict_choices["mh"], dict_choices["redshift"])),
        _fig,
        mo.md("The SEDs are shown in units of {:.0e} on the left-hand panel, and in units of {:.0e} on the right-hand panel.".format(_max_lum, _yy_max))])
    return


if __name__ == "__main__":
    app.run()
