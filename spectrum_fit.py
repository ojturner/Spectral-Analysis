#import the relevant modules
import os, sys, numpy as np, random, matplotlib.mlab as mlab, matplotlib.pyplot as plt, math, matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
from matplotlib.colors import LogNorm
import numpy.polynomial.polynomial as poly
import lmfit
from lmfit.models import GaussianModel, ExponentialModel, LorentzianModel, VoigtModel, PolynomialModel

#add my toolkit.py file to the PYTHONPATH
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
	
		

	table = pyfits.open(inFile)
	data = table[1].data
	flux = data.field('flux')
	wavelength = 10**(data.field('loglam'))
	ivar = data.field('ivar')
	weights = ivar
	redshift_data = table[2].data.field('Z')

	mod = PolynomialModel(6)
	pars = mod.guess(flux, x=wavelength)
	out  = mod.fit(flux, pars, x=wavelength)
	continuum = out.best_fit

	#At the moment get a crude estimate of the observed normalised SED for redshift computation
	normalised_observed_flux = flux / continuum

	#Call the normaliseTemplate method to find the normalised SED of a given template
	normalised_template_flux = normaliseTemplate('K20_late_composite_original.dat')
	plt.close('all')




	#Choose to plot the results 
	#plt.plot(wavelength, flux, label='flux')
	#plt.plot(wavelength, y_av, label='boxcar')
	#plt.plot(wavelength, ffit , label='fit')
	#legend()
	#plt.show()
	#plt.close('all')

	
	return {'flux' : flux, 'wavelength' : wavelength, 'z' : redshift_data, 'weights': weights}

##########################################################################################
#MODULE: templateRedshift
#
#PURPOSE:
#Take both the normalised template flux and crude normalised observed flux (or should 
#this be continuum subtracted?) and shift the template to find the redshift value
#the idea is to write this for a single template so that it can be looped around for many
#
#INPUTS:
#
#			wavelength: base wavelength array to plot against
#			t_flux: normalised template flux to 1  
#			flux: normalised observed flux to 1 		
#			
#
#OUTPUTS: 	z: Best redshift estimate from that template 
#
#USAGE: 	redshift = templateRedshift(wavelength, t_flux, flux)
#######################################################################################


def templateRedshift(wavelength, t_flux, flux):

	#Convert to numpy arrays
	wavelength = np.array(wavelength)
	t_flux = np.array(t_flux)
	flux = np.array(flux)

	#Now the t_flux and flux arrays will be of different lengths, 
	#append zeros to the shorter array 

	if (len(t_flux) - len(flux)) < 0:
		while len(t_flux < len(flux)):
			t_flux.append(1)

	else:
		while len(flux < len(t_flux)):
			flux.append(1)

	#So now both the flux and t_flux vectors are the same length hopefully		

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
	#This is the crude way to do it 
	mod = PolynomialModel(6)
	pars = mod.guess(flux, x=wavelength)
	out  = mod.fit(flux, pars, x=wavelength)
	continuum_poly = out.best_fit

	#Can also compute the continuum in the more advanced way
	#masking the emission lines and using a moving average


	#Define the wavelength values of the relevant emission lines
	OII3727 = 3727.092
	OII3729 = 3729.875
	H_beta = 4862.721
	OIII4959 = 4960.295
	OIII5007 = 5008.239
	H_alpha = 6564.614
	NII6585 = 6585.27
	SII6718 = 6718.29
	SII6732 = 6732.68

	#Now apply the redshift formula to find where this will be observed
	#Note that for these SDSS spectra the OII doublet is not in range
	OII3727_shifted = OII3727 * (1 + z)
	OII3729_shifted = OII3729 * (1 + z)
	H_beta_shifted = H_beta * (1 + z)
	OIII4959_shifted = OIII4959 * (1 + z)
	OIII5007_shifted = OIII5007 * (1 + z)
	H_alpha_shifted = H_alpha * (1 + z)
	NII6585_shifted = NII6585 * (1 + z)
	SII6718_shifted = SII6718 * (1 + z)
	SII6732_shifted = SII6732 * (1 + z)

	#hellofriend
	#Will choose to mask pm 15 for each of the lines
	H_beta_index = np.where(np.logical_and(wavelength>=(H_beta_shifted - 15), wavelength<=(H_beta_shifted + 15)))
	OIII_one_index = np.where(np.logical_and(wavelength>=(OIII4959_shifted - 15), wavelength<=(OIII4959_shifted + 15)))
	OIII_two_index = np.where(np.logical_and(wavelength>=(OIII5007_shifted - 15), wavelength<=(OIII5007_shifted + 15)))
	NII_one_index = np.where(np.logical_and(wavelength>=(NII6585_shifted - 15), wavelength<=(NII6585_shifted + 15)))
	H_alpha_index = np.where(np.logical_and(wavelength>=(H_alpha_shifted - 15), wavelength<=(H_alpha_shifted + 15)))
	SII_one_index = np.where(np.logical_and(wavelength>=(SII6718_shifted - 15), wavelength<=(SII6718_shifted + 15)))
	SII_two_index = np.where(np.logical_and(wavelength>=(SII6732_shifted - 15), wavelength<=(SII6732_shifted + 15)))

	#define the mask 1 values from the index values
	mask = np.zeros(len(flux))
	mask[H_beta_index] = 1
	mask[OIII_one_index] = 1
	mask[OIII_two_index] = 1
	mask[NII_one_index] = 1
	mask[H_alpha_index] = 1
	mask[SII_one_index] = 1
	mask[SII_two_index] = 1

	#Now apply these to the flux to mask 
	masked_flux = ma.masked_array(flux, mask=mask)

	#Make my own with np.mean()
	continuum = np.empty(len(masked_flux))
	for i in range(len(masked_flux)):

		if (i + 10) < len(masked_flux):
			continuum[i] = ma.median(masked_flux[i:i+5])
			if np.isnan(continuum[i]):
				continuum[i] = continuum[i - 1]
		else:
			continuum[i] = ma.median(masked_flux[i-5:i])
			if np.isnan(continuum[i]):
				continuum[i] = continuum[i - 1]

	

	#Subtract the continuum from the flux, just use polynomial fit right now 
	counts = flux - continuum

	#Construct a dictionary housing these shifted emission line values 
	#Note that values for the OII doublet are not present
	line_dict = {'H_beta' : H_beta_shifted, 'OIII4959' : OIII4959_shifted, 
	'OIII5007' : OIII5007_shifted, 'H_alpha' : H_alpha_shifted, 'NII6585' : NII6585_shifted, 
	'SII6718' : SII6718_shifted, 'SII6732' : SII6732_shifted}

	#Plot the initial continuum subtracted spectrum
	plt.plot(wavelength, counts)

	#Initialise a dictionary for the results in the for loop
	results_dict = {}

	print np.min(wavelength)
	#Begin for loop to fit an arbitrary number of emission lines
	for key in line_dict:
		
	
	########################################################################
	#FITTING EACH OF THE EMISSION LINES IN TURN
	########################################################################
	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength etc.

	#Use np.where to find the indices of data surrounding the gaussian
		new_index = np.where(np.logical_and(wavelength > (line_dict[key] - 10) ,
											wavelength < (line_dict[key] + 10)))  

	#Select only data for the fit with these indices
		fit_wavelength = wavelength[new_index]
		fit_counts = counts[new_index]
		fit_weights = weights[new_index]
		fit_continuum = continuum[new_index]

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

def galPhys(flux_Ha, flux_NeII6585, flux_OIII5007, flux_Hb):

	#Find the luminosity distance of the object at redshift z 
	#Convert the luminosity distance to cm  
	D_L = toolkit.cosmoDistanceQuad(z, 69.6, 0.286, 0.714)['D_L']
	D_L = D_L * 3.086E24

	#Compute H-alpha Luminosity in ergs / s
	Lum_Ha = 4 * np.pi * (D_L**2) * flux_Ha

	#Compute the star formation rate from the Kennicutt formula 
	SFR = np.log10(Lum_Ha) - 41.27

	#Compute the metallicity from the DO2 calibration
	Met = 9.12 + (0.73 * (flux_NeII6585/flux_Ha))

	OIII_ratio = np.log10((flux_OIII5007 * flux_Ha)/(flux_NeII6585 * flux_Hb))

	Met_2 = 8.73 - (0.32 * OIII_ratio)

	return {'SFR' : SFR, 'Metallicity_NII' : Met, 'Metallicity_OIII/NII' : Met_2}

##########################################################################################
#MODULE: normaliseTemplate
#
#PURPOSE:
#Take a file with template galaxy wavelength and flux values, computed from a galaxy SED code 
#and 
#
#INPUTS:
#
#			inFile: SED file containing flux and wavelength in columns 0 and 1
#
#
#OUTPUTS: 	Dictionary: containing the normalised flux, i.e. flux / continuum and the 
#						wavelength values in the spectral template
#
#
#USAGE: 	normalised_flux = normaliseTemplate(inFile)
###########################################################################################

def normaliseTemplate(inFile):

	#Read in the file and assign the flux and wavelength vectors
	template = np.loadtxt(inFile)
	wavelength = template[:,0]
	flux = template[:,1]

	#Plot the initial spectrum 
	plt.plot(wavelength, flux, color='green', label='template_spectrum')

	#Hardwire in the wavelength values
	OII = 3727
	H_beta = 4861 
	OIII_one = 4958
	OIII_two = 5006
	NII_one = 6548
	H_alpha = 6562
	NII_two = 6583
	SII_one = 6716
	SII_two = 6730

	#Will choose to mask pm 30 for each of the lines
	OII_index = np.where(np.logical_and(wavelength>=(OII - 15), wavelength<=(OII + 15)))
	H_beta_index = np.where(np.logical_and(wavelength>=(H_beta - 15), wavelength<=(H_beta + 15)))
	OIII_one_index = np.where(np.logical_and(wavelength>=(OIII_one - 15), wavelength<=(OIII_one + 15)))
	OIII_two_index = np.where(np.logical_and(wavelength>=(OIII_two - 15), wavelength<=(OIII_two + 15)))
	NII_one_index = np.where(np.logical_and(wavelength>=(NII_one - 15), wavelength<=(NII_one + 15)))
	H_alpha_index = np.where(np.logical_and(wavelength>=(H_alpha - 15), wavelength<=(H_alpha + 15)))
	NII_two_index = np.where(np.logical_and(wavelength>=(NII_two - 15), wavelength<=(NII_two + 15)))
	SII_one_index = np.where(np.logical_and(wavelength>=(SII_one - 15), wavelength<=(SII_one + 15)))
	SII_two_index = np.where(np.logical_and(wavelength>=(SII_two - 15), wavelength<=(SII_two + 15)))

	#define the mask 1 values from the index values
	mask = np.zeros(len(flux))
	mask[OII_index] = 1
	mask[H_beta_index] = 1
	mask[OIII_one_index] = 1
	mask[OIII_two_index] = 1
	mask[NII_one_index] = 1
	mask[H_alpha_index] = 1
	mask[NII_two_index] = 1
	mask[SII_one_index] = 1
	mask[SII_two_index] = 1

	#Now apply these to the flux to mask 
	masked_flux = ma.masked_array(flux, mask=mask)

	#Make my own with np.mean()
	continuum = np.empty(len(masked_flux))
	for i in range(len(masked_flux)):

		if (i + 10) < len(masked_flux):
			continuum[i] = ma.median(masked_flux[i:i+5])
			if np.isnan(continuum[i]):
				continuum[i] = continuum[i - 1]
		else:
			continuum[i] = ma.median(masked_flux[i-5:i])
			if np.isnan(continuum[i]):
				continuum[i] = continuum[i - 1]

	#Which works, (so long as I'm careful with masked values)
	#Now divide by the continuum and plot the final template

	normalised_flux = flux / continuum

	#Show off the plots
	plt.plot(wavelength, masked_flux, color='red', label='masked')
	plt.plot(wavelength, continuum, color='purple', linewidth=3, label='continuum')
	plt.plot(wavelength, normalised_flux, color='black', linewidth=2, label='normalised')

	return normalised_flux





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
	plt.show()

	#The given flux is in 10^-17 ergs / s / cm^2, grab from the fitLines results
	flux_Ha = fit_results['H_alpha'][0] * 1E-17 
	flux_Hb	= fit_results['H_beta'][0] * 1E-17
	flux_NeII6585 = fit_results['NII6585'][0] * 1E-17
	flux_OIII5007 = fit_results['OIII5007'][0] * 1E-17

	#Now use the galPhys method to compute the physical properties 
	props = galPhys(flux_Ha, flux_NeII6585, flux_OIII5007, flux_Hb)
	print props







	


	

	 
	


