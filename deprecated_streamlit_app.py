### Webtool to calculate cosmological K-corrections for star clusters
### Author: Marta Reina-Campos
### Date: September 2023

### Input: age and metallicity of the SSP, filter names
### For now: using the E-MILES stellar library of SEDs assuming BaSTI isochrones and Chabrier IMF, and the HST and JWST filter channels
### Output: a K-correction value

### Needs to be run: source activate science // streamlit run streamlit_app.py [-- script args]
### To create the requirements.txt file: python -m  pipreqs.pipreqs . --force

import streamlit as st
import os, glob, io, numpy, sys, matplotlib, pandas

from astropy.io import fits
from astropy import units as u
from astropy import constants
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, z_at_value
import astropy.cosmology.units as cu
from functions_Kcorrection import *
from scipy.interpolate import interp1d

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.size'] = 18.0
matplotlib.rcParams['legend.fontsize'] = 16.0

# custom CSS style
# global configuration
st.set_page_config(
   page_title="RESCUER: Cosmological K-corrections for star clusters",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
   menu_items={
        'Get Help': 'mailto:reinacampos@mcmaster.ca',
        'About': "## RESCUER: K-corrections for star clusters. \n Authors: Marta Reina-Campos and William E. Harris [September 2023]. Reference for the manuscript: "
    }
)

# function to select and load the required E-MILES model
def select_emiles_model(age, mh, name_slope, name_label, inpath):

    # format the name
    if mh < 0: sign_mh = "m"
    else: sign_mh = "p"
    name_mh = "Z{:s}{:.2f}".format(sign_mh, numpy.abs(mh))
    name_age = "T{:07.4f}".format(age)
        
    try:
        # load the FITS table
        fname_table = os.path.join(inpath, '{:s}{:s}{:s}_{:s}0.00_baseFe.fits'.format(name_slope, name_mh, name_age, name_label))
        key = "{:s}{:s}{:s}".format(name_slope, name_mh, name_age)
        hdul = fits.open(fname_table)
    except:
        st.write("{:s} could NOT be found".format(key))

    emiles_sed = {}

    # wavelength range for the E-MILES models - Å
    emiles_sed["lambda"] = numpy.linspace(1680.2, 49999.4, len(hdul[0].data)) * u.angstrom # Angstrom
    # transform to frequencies - Hz
    emiles_sed["nu"] = emiles_sed["lambda"].to(u.Hz, equivalencies=u.spectral()) # Hz

    ### LUMINOSITY
    # SSP spectra is output in Lλ/ LSun MSun^-1 Å^-1 units
    emiles_sed["lum_angstrom"] = hdul[0].data * u.solLum / u.angstrom / u.solMass
    # convert to erg/s/MSun/A
    emiles_sed["lum_angstrom"] = emiles_sed["lum_angstrom"].to(u.erg / u.s / u.angstrom / u.solMass)

    # transform to Hz^-1 -> f_nu = f_lambda*lambda/(nu = c/lambda)
    convert_to_hz = numpy.power(emiles_sed["lambda"], 2)/constants.c.to("angstrom s^-1")
    emiles_sed["lum_hz"] = emiles_sed["lum_angstrom"]*convert_to_hz / u.s / u.Hz
    hdul.close()

    return emiles_sed

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
    ls_files_filters = glob.glob(os.path.join(".", "Filters", "*.dat"))

    # output the data in a dictionary
    data_filter = {}
    # keep only the FXYZW
    range_filter = name_filter.split("_")[-1]
    # find the file
    for name in ls_files_filters:
        if range_filter == name.split(".")[-2]:
            tmp = numpy.genfromtxt(name, comments = "#")
            data_filter["lambda"] = tmp[:,0] * u.angstrom # Angstrom
            data_filter["nu"] = data_filter["lambda"].to(u.Hz, equivalencies=u.spectral()) # Hz
            data_filter["curve"] = tmp[:,1] # unitless
            break
    return data_filter  

# function to show the mulinosity of the chosen SSP
def plot_luminosities(age, mh, emiles_sed, z, obs_filter, rf_filter, target_sed, doing_bb = False):
    matplotlib.use('agg')  # required, use a non-interactive backend

    fig, axs = plt.subplots(2, figsize = (12,12.5), sharex = True, sharey = True, squeeze = False)
    axs = axs.flatten()

    left = 0.1; right = 0.87
    top = 0.98; bottom = 0.1
    hspace = 0.; wspace = 0.0

    max_lum = emiles_sed[target_sed]["lum_angstrom"].max()

    ### FIRST PANEL: homochromatic K-corrections 
    for i, ax in enumerate(axs):
        if i == 0:
            # add the intrinsic luminosities
            for key in emiles_sed.keys():
                if key == target_sed: 
                    color = "k"; lw = 1; ls = "-"; alpha = 1; label = "Intrinsic luminosity"
                    # add the redhsifted SED only for the target SED
                    ax.plot(emiles_sed[key]["lambda"].to("micron")*(1+z), emiles_sed[key]["lum_angstrom"]/max_lum/(1+z),
                        ls = ls, alpha = 0.3, c = "k", lw = lw, label = "Observed at z={:.1f}".format(z))
                    ax.plot(emiles_sed[key]["lambda"].to("micron"), emiles_sed[key]["lum_angstrom"]/max_lum,
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
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.tick_params(bottom = True, left= True, right = True, top = True, axis = "both", which = "both")
        ax.set_ylim(0, None)
        #ax.set_ylabel(r"$(1+z)^{{-1}}L_{{\lambda}}\left(\frac{{\lambda}}{{(1+z)}}\right)$ [{:.0e} ergs s$^{{-1}}$ $\AA^{{-1}}$ M$_{{\odot}}^{{-1}}$]".format(max_lum.value))     
        ax.set_ylabel("Normalised SEDs at a given distance")   
        
        if i == 0:
            ax.legend(loc = "center right", frameon = True)
        elif i == 1:
            ax.legend(loc = "upper right", frameon = True)

        ax.set_xlim(0, None)
        ax.minorticks_on()
        
    # format the entire figure
    fig.subplots_adjust(left = left, top = top, bottom = bottom, right = right, hspace = hspace, wspace = wspace)

    #buf = io.BytesIO()
    #fig.savefig(buf, bbox_inches = "tight")
    #fig_html = mpld3.fig_to_html(fig)
    #components.html(fig_html, height=600)
    st.pyplot(fig)
    #plt.close()

    return max_lum, yy_obs.max()

# function to check that the selected age and redshift are compatibles
def check_redshift_age_compatibility(age, z):
    # at what redshift they would be the oldest SSPs? - assumption: 500Myr post-BigBang for formation
    age += 0.5 # assumption: SSPs take at least 500Myr to start forming from the Big Bang
    z_oldest = z_at_value(Planck18.age, age * u.Gyr) 
    if z > z_oldest: return False
    else: return True

def plancks_law(wavelength, T):
    # wavelength in angstrom and T in K
    # spectral radiance (the power per unit solid angle and per unit of area normal to the propagation)
    # density of frequency nu  radiation per unit frequency at thermal equilibrium at temperature T
    # Units: power / [area × solid angle × frequency] -> power = energy/time.
    
    nu = constants.c.to("angstrom s^-1") / wavelength # 1/s = Hz   
    print(nu, T)
    spectral_radiance = (2.0 * constants.h.to("erg s") * nu*nu*nu)/(constants.c*constants.c*(numpy.exp(constants.h*nu/(constants.k_B*T)) - 1))
    
    # units: (ergs / s) sr^-1 Hz^-1
    return spectral_radiance.to("erg cm^-2") / u.s / u.Hz / u.sr

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

def main():
    ### PREPARE variables
    # define variables describing the E-MILES models
    #name_isochrone = "BASTI" # "PADOVA00"
    name_imf = "CH" # Chabrier 2003
    name_slope = "Ech1.30" # slope of the IMF

    dict_options = {"BASTI": {
                        "name_label" : "iTp",
                        "ls_models_mh" : numpy.asarray([-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25, 0.06, 0.15, 0.26, 0.4]),
                        "ls_models_ages" : numpy.asarray([0.0300, 0.0400, 0.0500, 0.0600, 0.0700, 0.0800, 0.0900, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 
                        0.3500, 0.4000, 0.4500, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.2500, 1.5000, 1.7500,
                        2.0000, 2.2500, 2.5000, 2.7500, 3.0000, 3.2500, 3.5000, 3.7500, 4.0000, 4.5000, 5.0000, 5.5000, 
                        6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000, 9.0000, 9.5000, 10.0000, 10.5000, 11.0000, 11.5000, 
                        12.0000, 12.5000, 13.0000]) # gyrs
                        },
                    "PADOVA00": {
                        "name_label" : "iPp",
                        "ls_models_mh" : numpy.asarray([-2.32, -1.71, -1.31, -0.71, -0.40, 0.00, 0.22]),
                        "ls_models_ages" : numpy.asarray([0.0631, 0.0708, 0.0794, 0.0891, 0.1000, 0.1122, 0.1259, 0.1413, 0.1585, 0.1778, 0.1995, 0.2239, 
                        0.2512, 0.2818, 0.3162, 0.3548, 0.3981, 0.4467, 0.5012, 0.5623, 0.6310, 0.7079, 0.7943, 0.8913, 1.0000, 1.1220,
                        1.2589, 1.4125, 1.5849, 1.7783, 1.9953, 2.2387, 2.5119, 2.8184, 3.1623, 3.5481, 3.9811, 4.4668, 5.0119, 5.6234,
                            6.3096, 7.0795, 7.9433, 8.9125, 10.0000, 11.2202, 12.5893]) # gyrs} 
                        }}
    
    ### load the available filters
    ls_files_filters = glob.glob(os.path.join(".", "Filters", "*.dat"))
    ls_filters = []
    for filt in ls_files_filters:
        ls_filters.append("_".join(filt.split("/")[-1].split(".")[:-1]))
    ls_filters.sort()

    ### SIDEBAR
    # place the selectboxes on the sidebar
    dict_choices = {}
    with st.sidebar:
        dict_choices["isochrone_model"] = st.selectbox(
                   "Select an isochrone model",
                   dict_options.keys(),
                   index=0,
                   placeholder="Select an isochrone model...",
                )
        # select an SED
        dict_choices["age"] = st.selectbox(
                   "Select an age for the SSP [Gyr]",
                   dict_options[dict_choices["isochrone_model"]]["ls_models_ages"],
                   index=34,
                   placeholder="Select an age...",
                )
        dict_choices["mh"] = st.selectbox(
                   "Select a metallicity for the SSP in [M/H]",
                   dict_options[dict_choices["isochrone_model"]]["ls_models_mh"],
                   index=0,
                   placeholder="Select an age...",
                )
        # place the SSP at a given redshift
        dict_choices["redshift"] = st.number_input("Redshift of the source [z in 0 - 1]", 
                                                    value=0.0, placeholder="Type a number...", 
                                                    min_value = 0.0, max_value = 1.0)

        # select the filters
        dict_choices["obs_filter"] = st.selectbox(
                   "Select the observed filter",
                   ls_filters,
                   index=0,
                   placeholder="Select a filter...",
                )
        dict_choices["rf_filter"] = st.selectbox(
                   "Select the rest-frame filter",
                   ls_filters,
                   index=0,
                   placeholder="Select a filter...",
                )
        
        # allow the user to input their desired uncertainties in age and metallicity
        dict_choices["add_uncertainties"] = st.checkbox("Do you want to add uncertainties?")
        if dict_choices["add_uncertainties"]:
            dict_choices["sigma_age"] = st.number_input("Relative age uncertainty [in %]", 
                                                    value=20.0, placeholder="Type a number...", 
                                                    min_value = 0.0, max_value = 100.0)
            dict_choices["sigma_mh"] = st.number_input("Metallicity uncertainty [in dex]", 
                                                    value=0.3, placeholder="Type a number...", 
                                                    min_value = 0.0, max_value = 3.0)

    ### START the webpage
    st.markdown('## RESCUER: Cosmological K-corrections for star clusters')
    #st.divider()
    #st.markdown('### Authors: Marta Reina-Campos and William E. Harris')
    st.markdown("Written by Marta Reina-Campos and William E. Harris. Based on the manuscript: [arXiv:2310.02307](https://ui.adsabs.harvard.edu/abs/2023arXiv231002307R/abstract)")

    # create the display
    st.markdown("This webtool uses the [E-MILES stellar library of SEDs](http://research.iac.es/proyecto/miles/pages/spectral-energy-distributions-seds/e-miles.php) assuming the {:s} models for the stellar isochrones and a Chabrier 2003 IMF.".format(dict_choices["isochrone_model"]))
    if dict_choices["add_uncertainties"]:
        st.markdown("The bounds in the K-correction are calculated using all the models contained within given user-defined uncertainties in age and metallicity, and they are given by the 10th - 90th percentiles of the K-corrections of these models.")

    # check whether the selected age and redshift can be physical
    compatible = check_redshift_age_compatibility(dict_choices["age"], dict_choices["redshift"])

    if compatible:

        # load the filter data
        obs_filter = load_filter_data(dict_choices["obs_filter"])
        rf_filter = load_filter_data(dict_choices["rf_filter"])

        # load the data
        inpath = os.path.join(os.curdir, "SEDs_E-MILES", "EMILES_{:s}_BASE_{:s}_FITS".format(dict_choices["isochrone_model"], name_imf))
        
        # define the target SED
        #target_sed = numpy.asarray([dict_choices["mh"], dict_choices["age"]])
        emiles_sed = {}
        # load the target SED
        name_mh = "Z{:.2f}".format(dict_choices["mh"])    
        name_age = "T{:07.4f}".format(dict_choices["age"])   
        key = "{:s}_{:s}".format(name_mh, name_age)            
        emiles_sed[key] = select_emiles_model(age = dict_choices["age"], mh = dict_choices["mh"],
                                        name_slope = name_slope, 
                                        name_label = dict_options[dict_choices["isochrone_model"]]["name_label"],
                                        inpath = inpath)
        dict_choices["target_sed"] = key
        # load the other models if needed 
        if dict_choices["add_uncertainties"]:
            # given the uncertainties, define the region and select the right models
            age_edges, mh_edges = define_edges_region(dict_choices, dict_options)
            subsample_models_age, subsample_models_mh = select_models_region(dict_choices, dict_options, age_edges, mh_edges)
            for age in subsample_models_age:
                name_age = "T{:07.4f}".format(age)   
                for mh in subsample_models_mh:
                    name_mh = "Z{:.2f}".format(mh)    
                    key = "{:s}_{:s}".format(name_mh, name_age)
                    if key not in emiles_sed.keys(): # if the model has not been loaded yet
                        emiles_sed[key] = select_emiles_model(age = age, mh = mh,
                                                    name_slope = name_slope, 
                                                    name_label = dict_options[dict_choices["isochrone_model"]]["name_label"],
                                                    inpath = inpath)
        dict_choices["list_seds"] = list(emiles_sed.keys())
        
        doing_blackbody = 0

        # if we're using the longest range filters, then overwrite the E-MILES SED and use a blackbody approximation
        if "F444W" in dict_choices["obs_filter"] or "F444W" in dict_choices["rf_filter"]:
            for key in emiles_sed.keys():
                # interpolate the E-MILES SED to get the peak wavelength
                interpolated = interp1d(emiles_sed[key]["lambda"], emiles_sed[key]["lum_angstrom"], axis = 0, fill_value = 'extrapolate')
                down_lambda = numpy.linspace(emiles_sed[key]["lambda"].min(), emiles_sed[key]["lambda"].max(), int(len(emiles_sed[key]["lambda"])/2))
                downsampled = interpolated(down_lambda)
                peak_lambda = down_lambda[ downsampled == downsampled.max()][0]
                # applying Wien's law to get the temperature
                T = ( 2898 * u.micron * u.K ).to(u.AA * u.K) / peak_lambda
                if key == dict_choices["target_sed"]:
                    doing_blackbody = T.value
                # resample the array containing the wavelengths
                max_wavelength = numpy.max([obs_filter["lambda"].max().value, rf_filter["lambda"].max().value])*rf_filter["lambda"].unit
                emiles_sed[key]["lambda"] = numpy.linspace(emiles_sed[key]["lambda"].min(), max_wavelength, len(emiles_sed[key]["lambda"]))
                # determine a blackbody spectrum - units: erg/s/sr/cm^2/Hz
                blackbody_nu = plancks_law(emiles_sed[key]["lambda"], T = T)
                # transforming quantities using nu*flux_nu = flux_lambda*lambda and lambda*nu = c
                blackbody_lambda = (blackbody_nu * u.Hz)*numpy.power(emiles_sed[key]["lambda"], -2)*constants.c.to("angstrom s^-1") * u.s
                # units: erg/s/sr/A
                emiles_sed[key]["lum_angstrom"] = 4*numpy.pi*numpy.power((10 * u.pc).to("cm"), 2)*blackbody_lambda
                
            st.markdown("**[WARNING]:** One of the selected filters exceeds the wavelength range covered by the E-MILES SEDs. The code is mimicking the chosen SED with a blackbody approximation of $T={:.1f}$ so it peaks at the same wavelength.".format(doing_blackbody))

        # define the standard source - AB magnitudes
        stdsource = define_standard_source(emiles_sed[dict_choices["target_sed"]])
        target_kcorr = 0; ls_kcorr = []
        for key in emiles_sed.keys():
            # calculate the K-correction
            kcorr = func_kcorrection_lambda(emiles_sed[key]["lambda"],
                                            emiles_sed[key]["lum_angstrom"],
                                            stdsource["lambda"],
                                            stdsource["flux_angstrom"], 
                                            obs_filter['lambda'], obs_filter['curve'],
                                            rf_filter['lambda'], rf_filter['curve'], dict_choices["redshift"])
            ls_kcorr.append(kcorr)
            if key == dict_choices["target_sed"]:
                target_kcorr = kcorr

            if dict_choices["add_uncertainties"]:
                # calculate the uncertainties as the distances to the 10th-90th percentiles
                per10th = numpy.min([numpy.percentile(ls_kcorr, 10), target_kcorr])
                per90th = numpy.max([numpy.percentile(ls_kcorr, 90), target_kcorr])
                dict_choices["sigma_kcorr"] = [per10th, per90th]
            
        # based on the value of the K-correction, choose an explanation
        if numpy.isnan(numpy.abs(target_kcorr)) or numpy.isinf(numpy.abs(target_kcorr)): target_kcorr = "--"; explanation = "**This SED and filter combination are misbehaving - check that the spectra is emitting in both wavelength ranges (i.e. that the SED covers the bandpasses of the filters).**"
        elif target_kcorr == -0.0: target_kcorr = 0; explanation = "**The intrinsic and observed SEDs are equal, make sure to enter a non-null redshift!**"
        elif target_kcorr < 0: explanation = "**[Interpretation]:** A negative K-correction indicates that the observed flux is brighter than the rest-frame SED, and thus the absolute magnitude should be dimmed."
        elif target_kcorr > 0: explanation = "**[Interpretation]:** A positive K-correction indicates that the observed flux is dimmer than the rest-frame SED, and thus the absolute magnitude should be brighter."
        elif target_kcorr == 0: explanation = "**[Interpretation]:** A null K-correction indicates that the observed flux is equal to the rest-frame SED."
        else: explanation = "Something went wrong with the K-correction, please try again."

        # format the label for the K-correction based on the above cases
        if type(target_kcorr) == str: 
            label_kcorr = "{:s}".format(target_kcorr)
            if dict_choices["add_uncertainties"]:
                label_uncertainties = "[lower] {:s} / [upper] {:s}".format(*dict_choices["sigma_kcorr"])
        else: 
            label_kcorr = "{:.5f}".format(target_kcorr, 5)
            if dict_choices["add_uncertainties"]:
                label_uncertainties = "[lower] {:.5f} / [upper] {:.5f}".format(*dict_choices["sigma_kcorr"])

        # create the table
        if doing_blackbody:
            dict_table = {'Age [Gyr]': ["Mimicking {:.1f} with a blackbody approximation".format(dict_choices["age"])],
            '[M/H]': ["Mimicking {:.1f} with a blackbody approximation".format(dict_choices["mh"])],
            'Redshift': [dict_choices["redshift"]],
            'Observed filter': [dict_choices["obs_filter"]],
            'Rest-frame filter': [dict_choices["rf_filter"]],
            'K-correction [AB mags]': [label_kcorr]}
            if dict_choices["add_uncertainties"]:
                dict_table['Bounds in the K-correction [AB mags]'] = [label_uncertainties]

        else:
            dict_table = {'Age [Gyr]': [dict_choices["age"]],
            '[M/H]': [dict_choices["mh"]],
            'Redshift': [dict_choices["redshift"]],
            'Observed filter': [dict_choices["obs_filter"]],
            'Rest-frame filter': [dict_choices["rf_filter"]],
            'K-correction [AB mags]': [label_kcorr]}
            if dict_choices["add_uncertainties"]:
                dict_table['Bounds in the K-correction [AB mags]'] = [label_uncertainties]

        df = pandas.DataFrame.from_dict(dict_table, orient = "index")

        # divide in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**[Table]:** Summary of the parameters chosen for the SSP and the filters. The K-correction is given in AB mags.")
            #st.table(dict_table)
            st.dataframe(df, use_container_width = True, column_config = {"_index" : "Parameters", "0" : "Values"})

            #if dict_choices["add_uncertainties"]:
            #    st.markdown("**[Explanation]:** The uncertainties in age and metallicity are {:.2f} Gyr and {:.2f} dex, respectively (they need to be readjusted to the spacing of the grid of models). The K-correction has an uncertainty of - {:.5f} / + {:.5f} AB mags.".format(dict_choices["sigma_age"]*dict_choices["age"]/100, dict_choices["sigma_mh"], *dict_choices["sigma_kcorr"]))

            st.markdown(explanation)
            # , with (grey) lines showing the SEDs of the models used to calculate the uncertainty in the K-correction
            st.markdown("**[Image]:** Expected behaviour of the K-correction on the SED of a SSP of {:.2f} Gyr and [M/H] = {:.2f} observed at z = {:.2f}: (top) Intrinsic luminosity of the SSP from E-MILES (black) and its redshifted curve (grey) as a function of wavelength, and (bottom) behaviour of the K-correction. The vertical shaded areas indicate the wavelength range covered by each filter. The correction given by the K value is indicated by the red arrow.".format(dict_choices["age"], dict_choices["mh"], dict_choices["redshift"]))

        with col2:
            # plot the luminosities and K - correction
            max_lum, yy_max = plot_luminosities(dict_choices["age"], dict_choices["mh"], emiles_sed, 
                            z = dict_choices["redshift"], obs_filter = obs_filter, rf_filter = rf_filter,
                            target_sed = dict_choices["target_sed"], doing_bb = doing_blackbody)

            st.markdown("The SEDs are shown in units of {:.0e} on the upper panel, and in units of {:.0e} on the lower panel.".format(max_lum, yy_max))

    else:
        max_age = Planck18.age(dict_choices["redshift"]*cu.redshift).to(u.Gyr) - 0.5*u.Gyr
        st.markdown("**The adopted age is not measurable at z={:.3f} in the Planck18 cosmology. The maximum allowed age at this redshift is {:.2f} (assuming 500 Myr for formation after the Big Bang). Please try again.**".format(dict_choices["redshift"], max_age))

if __name__ == '__main__':
    main() # run the main function

#pywebio.start_server(main, port=8080)