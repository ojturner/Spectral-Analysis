#import the relevant modules
import os, sys, numpy as np, random, matplotlib.mlab as mlab, matplotlib.pyplot as plt, math, matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
from matplotlib.colors import LogNorm
import numpy.polynomial.polynomial as poly
import lmfit
from lmfit.models import GaussianModel, ExponentialModel, LorentzianModel, VoigtModel, PolynomialModel

#temporarily add my toolkit.py file to the PYTHONPATH
sys.path.append('/Users/owenturner/Documents/PhD/SUPA_Courses/AdvancedDataAnalysis/Homeworks')

#and import the toolkit
import toolkit
import pyfits

######################################################################################
#MODULE: readFits
#
#PURPOSE:
#Read in these SDSS fits files and find the flux and wavelength
#
#
#INPUTS:
#
#			inFile: full name of a fits file in the current directory
#					Note this is currently looped externally to the module
#
#
#OUTPUTS: 	dictionary: with keys 'flux', 'wavelength', 'z', 'continuum', 'error'
#						corresponding to the attributes extracted from the fits file
#						
#
#USAGE: 	results_dict = readFits(inFile)
#######################################################################################


def readFits(inFile): 
	#import the relevant modules
		

	table = pyfits.open(name)
	data = table[1].data
	flux = data.field('flux')
	wavelength = 10**(data.field('loglam'))
	ivar = data.field('ivar')
	weights = ivar
	redshift_data = table[2].data.field('Z')
	#y_av = toolkit.movingaverage(flux, 50)
	#coefs = poly.polyfit(wavelength, flux, 8, w=ivar)
	#ffit = poly.polyval(wavelength, coefs)

	#print(out.fit_report(min_correl=0.25))
	#plt.plot(wavelength, out.best_fit, 'r-', linewidth=2)

	#Choose to plot the results 
	#plt.plot(wavelength, flux, label='flux')
	#plt.plot(wavelength, y_av, label='boxcar')
	#plt.plot(wavelength, ffit , label='fit')
	#legend()
	#plt.show()
	#plt.close('all')

	
	return {'flux' : flux, 'wavelength' : wavelength, 'z' : redshift_data, 'weights': weights}



##########################################################################################
#MODULE: fitLines
#
#PURPOSE:
#Use the results of the readFits method to first model and subtract the continuum emission 
#and then fit gaussians to the H-alpha, H-beta, OIII4959, OIII5007 emission lines
#so that the flux of these emission lines can be measured.
#
#INPUTS:
#
#			flux: The full array of photon counts read in from a fits file
#			wavelength: The corresponding wavelength values of the counts 
#			z: The redshift of the source 		
#			weights: Individual inverse variance weights on the fluxes
#
#OUTPUTS: 	dictionary: with keys corresponding to the names of the emission lines 
#						each key contains 3 elements for the amplitude, sigma and fwhm
#						of the fitted gaussians
#
#USAGE: 	results_dict = fitLines(flux, wavelength, z, weights)
#######################################################################################


def fitLines(flux, wavelength, z, weights): 	

	#Convert all into numpy arrays 
	flux = np.array(flux)
	wavelength = np.array(wavelength)
	z = np.array(z)
	weights = np.array(weights)

	#Fit a polynomial to the continuum background emission of the galaxy
	mod = PolynomialModel(6)
	pars = mod.guess(flux, x=wavelength)
	out  = mod.fit(flux, pars, x=wavelength)
	continuum = out.best_fit

	#Subtract the continuum from the flux, just use polynomial fit right now 
	counts = flux - continuum

	#Define the wavelength values of the relevant emission lines
	H_beta = 4862.721
	H_alpha = 6564.614
	OIII4959 = 4960.295
	OIII5007 = 5008.239
	NII6585 = 6585.27

	#Now apply the redshift formula to find where this will be observed
	H_alpha_shifted = H_alpha * (1 + z)
	H_beta_shifted = H_beta * (1 + z)
	OIII4959_shifted = OIII4959 * (1 + z)
	OIII5007_shifted = OIII5007 * (1 + z)
	NII6585_shifted = NII6585 * (1 + z)

	#Construct a dictionary housing these shifted emission line values 
	line_dict = {'H_alpha' : H_alpha_shifted, 'H_beta' : H_beta_shifted, 
	'OIII4959' : OIII4959_shifted, 'OIII5007' : OIII5007_shifted, 'NII6585' : NII6585_shifted}

	#Plot the initial continuum subtracted spectrum
	plt.plot(wavelength, counts)

	#Initialise a dictionary for the results in the for loop
	results_dict = {}

	#Begin for loop to fit an arbitrary number of emission lines
	for key in line_dict:
	
	########################################################################
	#FITTING EACH OF THE EMISSION LINES IN TURN
	########################################################################
	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength etc.

	#Use np.where to find the indices of data surrounding the gaussian
		index = np.where( wavelength > (line_dict[key] - 10) )
		new_wavelength = wavelength[index]
		new_counts = counts[index]
		new_weights = weights[index]
		new_continuum = continuum[index]
		new_index = np.where( new_wavelength < (line_dict[key] + 10))

	#Select only data for the fit with these indices
		fit_wavelength = new_wavelength[new_index]
		fit_counts = new_counts[new_index]
		fit_weights = new_weights[new_index]
		fit_continuum = new_continuum[new_index]

	#Compute the equivalent width
		edge_continuum = fit_counts[0]
		delta_lambda = abs(fit_wavelength[1] - fit_wavelength[0])	
		E_w = sum(abs(1 - (fit_counts/edge_continuum))) * delta_lambda

	#Now use the lmfit package to perform gaussian fits to the data	
	#Construct the gaussian model
		mod = GaussianModel()

	#Take an initial guess at what the model parameters are 
	#In this case the gaussian model has three parameters, 
	#Which are amplitude, center and sigma
		pars = mod.guess(fit_counts, x=fit_wavelength)

	#We know from the redshift what the center of the gaussian is, set this
	#And choose the option not to vary this parameter 
	#Leave the guessed values of the other parameters
		pars['center'].set(value = line_dict[key])
		pars['center'].set(vary = 'False')

	#Now perform the fit to the data using the set and guessed parameters 
	#And the inverse variance weights form the fits file 
		out  = mod.fit(fit_counts, pars, weights = fit_weights, x=fit_wavelength)
		#print(out.fit_report(min_correl=0.25))

	#Plot the results and the spectrum to check the fit
		plt.plot(fit_wavelength, out.best_fit, 'r-')
	

	#The amplitude parameter is the area under the curve, equivalent to the flux
		results_dict[key] = [out.best_values['amplitude'], out.best_values['sigma'], 2.3548200*out.best_values['sigma'], E_w]
	

	#The return dictionary for this method is a sequence of results vectors
	return results_dict

##########################################################################################
#MODULE: galPhys
#
#PURPOSE:
#Use the results of the fitLines method to calculate physical quantities for the galaxies 
#
#INPUTS:
#
#			flux_Ha: In units of ergs / s / cm^2
#			flux_NeII6585: In units of ergs / s / cm^2
#			z: redshift of the source
#
#
#OUTPUTS: 	dictionary: with keys corresponding to the SFR and metallicity of the galaxy 
#						Note that the metallicity is in units 12 + log(O/H) and the 
#						SFR is in units of log(Msun/yr)
#
#
#USAGE: 	results_dict = galPhys(flux_Ha, flux_NeII6585)
###########################################################################################

def galPhys(flux_Ha, flux_NeII6585):

	#Find the luminosity distance of the object at redshift z 
	#Convert the luminosity distance to cm  
	D_L = toolkit.cosmoDistanceQuad(z, 69.6, 0.286, 0.714)['D_L']
	D_L = D_L * 3.086E26

	#Compute H-alpha Luminosity in ergs / s
	Lum_Ha = 4 * np.pi * (D_L**2) * flux_Ha

	#Compute the star formation rate from the Kennicutt formula 
	SFR = np.log10(Lum_Ha) - 41.27

	#Compute the metallicity from the DO2 calibration
	Met = 9.12 + (0.73 * (flux_NeII6585/flux_Ha))

	return {'SFR' : SFR, 'Metallicity' : Met}






#Read the names of the fits files in 
data=np.genfromtxt('names.txt', dtype=None)

#Loop over each name and apply the defined methods to each of them
#Ultimately this is computing the physical properties of the galaxies
for name in data:

	#Read the fits files and define variables from the resulting dictionary
	dict_results = readFits(name)
	flux = dict_results['flux']		
	wavelength = dict_results['wavelength']
	z = dict_results['z']
	weights = dict_results['weights']

	#Fit the emission lines using the defined method, plot the results
	fit_results = fitLines(flux, wavelength, z, weights)
	#print fit_results

	#Now show the plots if we want
	#plt.show()

	#The given flux is in 10^-17 ergs / s / cm^2, grab from the fitLines results
	flux_Ha = fit_results['H_alpha'][0] * 1E-17 
	flux_NeII6585 = fit_results['NII6585'][0] * 1E-17

	#Now use the galPhys method to compute the physical properties 
	props = galPhys(flux_Ha, flux_NeII6585)
	print props



	


	

	 
	


