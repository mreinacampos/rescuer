### Script to calculate the K-correction needed on a given SED, for a source at redshift z observed in filter Q and emitted at filter R
### Author: Marta Reina-Campos
### Date: July 2023
### Following the equation on Hogg+02 and Reina-Campos&Harris 2023

# Import modules
import sys, numpy, os, glob, scipy
from scipy import interpolate
from astropy.io import fits
from astropy import units as u
from astropy import constants

# define the physical types of certain unit combinations
#u.physical.def_physical_type(u.erg * u.s**-1 * u.Angstrom**-1 * u.sr**-1, "luminosity")
#u.physical.def_physical_type(u.erg * u.s**-1 * u.Angstrom**-1, "luminosity")
#u.physical.def_physical_type(u.erg * u.s**-1 * u.Angstrom**-1 * u.cm**-2, "flux")
#u.physical.def_physical_type(u.erg * u.s**-1 * u.Angstrom**-1 * u.cm**-2 * u.sr**-1, "flux")

### Integrals over frequency space

@u.quantity_input
def func_integral_flux_nu(nu: u.Quantity[u.Hz], 
                          model_nu : u.Quantity[u.Hz], 
                          model_flux,# : #"flux",
                          bandpass_nu : u.Quantity[u.Hz],
                          bandpass_curve, mode = "emitted", z = 0):
    # function to calculate the integral over the spectral flux density [units: ergs/s/cm^2/Hz] in frequency space
    # input parameters:
        # nu: frequency [Hz] range to integrate over
        # model: frequency [Hz] and spectral flux density [ergs/s/cm^2/Hz] of the model 
        # bandpass: filter response [unitless]
        # mode [emitted/observed] : whether we're calculating the integral over the emitted or observed ranges
        # z: redshift of the source
    
    # make sure the input parameters are in the right units
    nu = nu.to(u.Hz)
    model_nu = model_nu.to(u.Hz)
    bandpass_nu = bandpass_nu.to(u.Hz)
    
    # interpolate the spectral flux density - units: erg/s/cm^2/Hz 
    intrp_func_flux = scipy.interpolate.interp1d(model_nu, model_flux, kind='cubic')
    
    # we need to transform the frequency range whether we're calculating it over the emitted or observed ranges
    if "obs" in mode.lower():
        intrp_flux = intrp_func_flux(nu) * model_lum.unit
    elif "emi" in mode.lower():
        intrp_flux = intrp_func_flux(nu/(1+z)) * model_lum.unit
        
    # interpolate the bandpass of the filter
    intrp_func_bandpass = scipy.interpolate.interp1d(bandpass_nu, bandpass_curve, kind='cubic')
    intrp_bandpass = intrp_func_bandpass(nu)

    # integrate over the spectral flux density - units: erg/s/cm^2/Hz 
    int_flux = numpy.sum((1/nu)*intrp_flux*intrp_bandpass)

    return int_flux

@u.quantity_input
def func_integral_luminosity_nu(nu: u.Quantity[u.Hz], 
                                model_nu : u.Quantity[u.Hz], 
                                model_lum,# : "luminosity",
                                bandpass_nu : u.Quantity[u.Hz],
                                bandpass_curve, mode = "emitted", z = 0):
    # function to calculate the integral over the intrinsic luminosity [units: ergs/s/Hz] in frequency space
    # input parameters:
        # nu: frequency [Hz] range to integrate over
        # model: frequency [Hz] and intrinsic luminosity [ergs/s/Hz] of the model 
        # bandpass: filter response [unitless]
        # mode [emitted/observed] : whether we're calculating the integral over the emitted or observed ranges
        # z: redshift of the source
        
    # make sure the input parameters are in the right units
    nu = nu.to(u.Hz)
    model_nu = model_nu.to(u.Hz)
    bandpass_nu = bandpass_nu.to(u.Hz)

    # interpolate the intrinsic luminosity - units: erg/s/Hz 
    intrp_func_lum = scipy.interpolate.interp1d(model_nu, model_lum, kind='cubic')
    
    # we need to transform the frequency range whether we're calculating it over the emitted or observed ranges
    if "obs" in mode.lower():
        intrp_lum = intrp_func_lum(nu*(1+z)) * model_lum.unit
    elif "emi" in mode.lower():
        intrp_lum = intrp_func_lum(nu) * model_lum.unit
        
    # interpolate the bandpass of the filter
    intrp_func_bandpass = scipy.interpolate.interp1d(bandpass_nu, bandpass_curve, kind='cubic')
    intrp_bandpass = intrp_func_bandpass(nu)

    # integrate over the intrinsic luminosity - units: erg/s/Hz 
    int_lum = numpy.sum((1/nu)*intrp_lum*intrp_bandpass)

    return int_lum

@u.quantity_input
def func_integral_stdsource_nu(nu: u.Quantity[u.Hz], 
                               stdsource_nu : u.Quantity[u.Hz], 
                               stdsource_flux,# : "flux",
                               bandpass_nu : u.Quantity[u.Hz],
                               bandpass_curve):
    # function to calculate the integral over the standard source [units: ergs/s/cm^2/Hz] in frequency space
    # input parameters:
        # nu: frequency [Hz] range to integrate over
        # model: frequency [Hz] and spectral flux density [ergs/s/Hz/cm^2] of the standard source 
        # bandpass: filter response [unitless]
    
    # make sure the input parameters are in the right units
    nu = nu.to(u.Hz)
    stdsource_nu = stdsource_nu.to(u.Hz)
    bandpass_nu = bandpass_nu.to(u.Hz)

    # interpolate the std source density - units: erg/s/Hz/cm^2
    intrp_func_den = scipy.interpolate.interp1d(stdsource_nu, stdsource_flux, kind='cubic')
    intrp_den = intrp_func_den(nu) * stdsource_flux.unit
        
    # interpolate the bandpass of the filter
    intrp_func_bandpass = scipy.interpolate.interp1d(bandpass_nu, bandpass_curve, kind='cubic')
    intrp_bandpass = intrp_func_bandpass(nu)

    # integrate over the std source density - units: erg/s/Hz/cm^2
    int_stdsource = numpy.sum((1/nu)*intrp_den*intrp_bandpass)

    return int_stdsource

### Integrals over wavelength space

@u.quantity_input
def func_integral_flux_lambda(wavelength: u.Quantity[u.Angstrom], 
                              model_lambda : u.Quantity[u.Angstrom], 
                              model_flux,# : "flux",
                              bandpass_lambda : u.Quantity[u.Angstrom],
                              bandpass_curve, mode = "emitted", z = 0):
    # function to calculate the integral over the spectral flux density [units: ergs/s/cm^2/A] in wavelength space
    # input parameters:
        # wavelength: wavelength [A] range to integrate over
        # model: wavelength [A] and spectral flux density [ergs/s/cm^2/A] of the model 
        # bandpass: filter response [unitless]
        # mode [emitted/observed] : whether we're calculating the integral over the emitted or observed ranges
        # z: redshift of the source
    
    # make sure the input parameters are in the right units
    wavelength = wavelength.to(u.Angstrom)
    model_lambda = model_lambda.to(u.Angstrom)
    bandpass_lambda = bandpass_lambda.to(u.Angstrom)
 
    # interpolate the spectral flux density - units: erg/s/cm^2/A 
    intrp_func_flux = scipy.interpolate.interp1d(model_lambda, model_flux, kind='cubic')
    
    # we need to transform the frequency range whether we're calculating it over the emitted or observed ranges
    if "obs" in mode.lower():
        intrp_flux = intrp_func_flux(wavelength) * model_lum.unit
    elif "emi" in mode.lower():
        intrp_flux = intrp_func_flux(wavelength*(1+z)) * model_lum.unit
        
    # interpolate the bandpass of the filter
    intrp_func_bandpass = scipy.interpolate.interp1d(bandpass_lambda, bandpass_curve, kind='cubic')
    intrp_bandpass = intrp_func_bandpass(wavelength)

    # integrate over the spectral flux density - units: erg/s/cm^2/A
    int_flux = numpy.sum(wavelength*intrp_flux*intrp_bandpass)

    return int_flux

@u.quantity_input
def func_integral_luminosity_lambda(wavelength: u.Quantity[u.Angstrom], 
                                    model_lambda : u.Quantity[u.Angstrom], 
                                    model_lum,# : "luminosity",
                                    bandpass_lambda : u.Quantity[u.Angstrom],
                                    bandpass_curve, mode = "emitted", z = 0):
    # function to calculate the integral over the intrinsic luminosity [units: ergs/s/A] in wavelength space
    # input parameters:
        # wavelength: wavelength [A] range to integrate over
        # model: wavelength [A] and intrinsic luminosity [ergs/s/A] of the model 
        # bandpass: filter response [unitless]
        # mode [emitted/observed] : whether we're calculating the integral over the emitted or observed ranges
        # z: redshift of the source
    
    # make sure the input parameters are in the right units
    wavelength = wavelength.to(u.Angstrom)
    model_lambda = model_lambda.to(u.Angstrom)
    bandpass_lambda = bandpass_lambda.to(u.Angstrom)

    # interpolate the intrinsic luminosity - units: erg/s/Hz 
    intrp_func_lum = scipy.interpolate.interp1d(model_lambda, model_lum, kind='cubic')
        
    # we need to transform the frequency range whether we're calculating it over the emitted or observed ranges
    if "obs" in mode.lower():
        intrp_lum = intrp_func_lum(wavelength/(1+z)) * model_lum.unit
    elif "emi" in mode.lower():
        intrp_lum = intrp_func_lum(wavelength) * model_lum.unit
        
    # interpolate the bandpass of the filter
    intrp_func_bandpass = scipy.interpolate.interp1d(bandpass_lambda, bandpass_curve, kind='cubic')
    intrp_bandpass = intrp_func_bandpass(wavelength) 

    # integrate over the intrinsic luminosity - units: erg/s/Hz 
    int_lum = numpy.sum(wavelength*intrp_lum*intrp_bandpass)

    return int_lum

@u.quantity_input
def func_integral_stdsource_lambda(wavelength: u.Quantity[u.Angstrom], 
                                   stdsource_lambda : u.Quantity[u.Angstrom], 
                                   stdsource_flux,# : "flux",
                                   bandpass_lambda : u.Quantity[u.Angstrom],
                                   bandpass_curve):
    # function to calculate the integral over the standard source [units: ergs/s/cm^2/A] in wavelength space
    # input parameters:
        # wavelength: wavelength [A] range to integrate over
        # model: wavelength [A] and spectral flux density [ergs/s/A/cm^2] of the standard source 
        # bandpass: filter response [unitless]
    
    # make sure the input parameters are in the right units
    wavelength = wavelength.to(u.Angstrom)
    stdsource_lambda = stdsource_lambda.to(u.Angstrom)
    bandpass_lambda = bandpass_lambda.to(u.Angstrom)
 
    # interpolate the std source density - units: erg/s/A/cm^2
    intrp_func_den = scipy.interpolate.interp1d(stdsource_lambda, stdsource_flux, kind='cubic')
    intrp_den = intrp_func_den(wavelength) * stdsource_flux.unit

    # interpolate the bandpass of the filter
    intrp_func_bandpass = scipy.interpolate.interp1d(bandpass_lambda, bandpass_curve, kind='cubic')
    intrp_bandpass = intrp_func_bandpass(wavelength)

    # integrate over the std source density - units: erg/s/A/cm^2
    int_stdsource = numpy.sum(wavelength*intrp_den*intrp_bandpass)

    return int_stdsource

### K-correction in frequency space

@u.quantity_input
def func_kcorrection_nu(table_model_nu: u.Quantity[u.Hz], # frequencies of the SED
                        table_model_sed, # SED: luminosity or flux
                        stdsource_nu: u.Quantity[u.Hz], # frequencies of the standard source
                        stdsource_flux, # flux of the standard source
                        obs_bandpass_nu : u.Quantity[u.Hz], obs_bandpass_curve,
                        em_bandpass_nu: u.Quantity[u.Hz], em_bandpass_curve, # observed and emitted bandpasses
                        z):
    
    # Calculating the K-correction integrating the intrinsic luminosity or the spectral flux over frequencies
    # k-correction in frequency space - eq. (8 and 9) in Hoggs+02

    # convert to the right units
    obs_bandpass_nu = obs_bandpass_nu.to(u.Hz)
    em_bandpass_nu = em_bandpass_nu.to(u.Hz)
    table_model_nu = table_model_nu.to(u.Hz)
    stdsource_nu = stdsource_nu.to(u.Hz)
    
    # FIRST: define the range of frequencies to integrate over
    obs_nu = numpy.linspace(obs_bandpass_nu.min(), obs_bandpass_nu.max(), 601) 
    em_nu = numpy.linspace(em_bandpass_nu.min(), em_bandpass_nu.max(), 601) 
 
    # SECOND: integrate over the luminosity or flux
    if "Hz" in table_model_sed.unit.to_string():
        if "cm^-2" in table_model_sed.unit.to_string(): # if the SED is a spectral flux
            obs_int_sed = func_integral_flux_nu(obs_nu, 
                                                table_model_nu, table_model_sed,
                                                obs_bandpass_nu, obs_bandpass_curve, mode = "observed", z = z)
            em_int_sed = func_integral_flux_nu(em_nu, 
                                               table_model_nu, table_model_sed, 
                                               em_bandpass_nu, em_bandpass_curve, mode = "emitted", z = z)
        else: # else if it is an intrinsic luminosity
            obs_int_sed = func_integral_luminosity_nu(obs_nu, 
                                                      table_model_nu, table_model_sed, 
                                                      obs_bandpass_nu, obs_bandpass_curve, mode = "observed", z = z)
            em_int_sed = func_integral_luminosity_nu(em_nu, 
                                                     table_model_nu, table_model_sed, 
                                                     em_bandpass_nu, em_bandpass_curve, mode = "emitted", z = z)
    else:
        print("The provided SED is neither a flux nor a luminosity! - Units: {:s}".format(table_model_sed.unit) )
    
    # THIRD: integrate over the standard soure
    obs_int_stdsource = func_integral_stdsource_nu(obs_nu, 
                                                   stdsource_nu, stdsource_flux, 
                                                   obs_bandpass_nu, obs_bandpass_curve)
    em_int_stdsource = func_integral_stdsource_nu(em_nu, 
                                                  stdsource_nu, stdsource_flux, 
                                                  em_bandpass_nu, em_bandpass_curve)
    
    ### THIRD: K-correction
    kcorrection = -2.5*numpy.log10((1+z)*(obs_int_sed*em_int_stdsource)/(obs_int_stdsource*em_int_sed))

    return kcorrection


### K-correction in wavelength space
def func_kcorrection_lambda(table_model_lambda: u.Quantity[u.Angstrom], # frequencies of the SED
                            table_model_sed, # SED: luminosity or flux
                            stdsource_lambda: u.Quantity[u.Angstrom], # frequencies of the standard source
                            stdsource_flux, # flux of the standard source
                            obs_bandpass_lambda : u.Quantity[u.Angstrom], obs_bandpass_curve,
                            em_bandpass_lambda: u.Quantity[u.Angstrom], em_bandpass_curve, # observed and emitted bandpasses
                            z):
    
    # Calculating the K-correction integrating the intrinsic luminosity or the spectral flux over wavelengths
    # k-correction in wavelength space - eq. (12 and 13) in Hoggs+02

    # convert to the right units
    obs_bandpass_lambda = obs_bandpass_lambda.to(u.Angstrom)
    em_bandpass_lambda = em_bandpass_lambda.to(u.Angstrom)
    table_model_lambda = table_model_lambda.to(u.Angstrom)
    stdsource_lambda = stdsource_lambda.to(u.Angstrom)
    
    # FIRST: define the range of wavelengths to integrate over
    obs_lambda = numpy.linspace(obs_bandpass_lambda.min(), obs_bandpass_lambda.max(), 601) 
    em_lambda = numpy.linspace(em_bandpass_lambda.min(), em_bandpass_lambda.max(), 601) 

    # SECOND: integrate over the luminosity or flux
    if "Angstrom" in table_model_sed.unit.to_string():
        if "cm^-2" in table_model_sed.unit.to_string(): # if the SED is a spectral flux
            obs_int_sed = func_integral_flux_lambda(obs_lambda, 
                                                table_model_lambda, table_model_sed,
                                                obs_bandpass_lambda, obs_bandpass_curve, mode = "observed", z = z)
            em_int_sed = func_integral_flux_lambda(em_lambda, 
                                               table_model_lambda, table_model_sed, 
                                               em_bandpass_lambda, em_bandpass_curve, mode = "emitted", z = z)
        else: # if the SED is a luminosity
            obs_int_sed = func_integral_luminosity_lambda(obs_lambda, 
                                                      table_model_lambda, table_model_sed, 
                                                      obs_bandpass_lambda, obs_bandpass_curve, mode = "observed", z = z)
            em_int_sed = func_integral_luminosity_lambda(em_lambda, 
                                                     table_model_lambda, table_model_sed, 
                                                     em_bandpass_lambda, em_bandpass_curve, mode = "emitted", z = z)
    else:
        print("The provided SED is neither a flux nor a luminosity! - Units: {:s}".format(table_model_sed.unit) )
    
    # THIRD: integrate over the standard soure
    obs_int_stdsource = func_integral_stdsource_lambda(obs_lambda, 
                                                   stdsource_lambda, stdsource_flux, 
                                                   obs_bandpass_lambda, obs_bandpass_curve)
    em_int_stdsource = func_integral_stdsource_lambda(em_lambda, 
                                                  stdsource_lambda, stdsource_flux, 
                                                  em_bandpass_lambda, em_bandpass_curve)
    
    ### THIRD: K-correction
    kcorrection = -2.5*numpy.log10((1/(1+z))*(obs_int_sed*em_int_stdsource)/(obs_int_stdsource*em_int_sed))
    
    return kcorrection

