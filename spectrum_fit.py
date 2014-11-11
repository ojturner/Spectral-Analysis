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


#read in these SDSS fits files and find the flux and wavelength
#Also fit the continuum emission from the galaxy
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

	mod2 = PolynomialModel(6)
	pars = mod2.guess(flux, x=wavelength)
	out  = mod2.fit(flux, pars, x=wavelength)
	#print(out.fit_report(min_correl=0.25))
	#plt.plot(wavelength, out.best_fit, 'r-', linewidth=2)

		#Choose to plot the results 
	#plt.plot(wavelength, flux, label='flux')
	#plt.plot(wavelength, y_av, label='boxcar')
	#plt.plot(wavelength, ffit , label='fit')
	#legend()
	#plt.show()
	#plt.close('all')

	
	return {'flux' : flux, 'wavelength' : wavelength, 'z' : redshift_data, 'continuum': out.best_fit, 'error': weights}

def fitLines(flux, wavelength, z, continuum, error): 	

#First attempt fitting a gaussian with all the data before masking off most of it
#Will first fit the H-alpha line so define the wavelength of H-alpha (angstroms)
#################################################################
#POSSIBLY THESE SHOULD INVOLVE A VACUUM CORRECTION 
#################################################################

	#Convert all into numpy arrays 
	flux = np.array(flux)
	wavelength = np.array(wavelength)
	z = np.array(z)
	continuum = np.array(continuum)
	error = np.array(error)

	H_beta = 4862.721
	H_alpha = 6564.614
	OIIIfirst = 4960.295
	OIIIsecond = 5008.239

	#Now apply the redshift formula to find where this will be observed
	H_alpha_shifted = H_alpha * (1 + z)
	H_beta_shifted = H_beta * (1 + z)
	OIIIfirst_shifted = OIIIfirst * (1 + z)
	OIIIsecond_shifted = OIIIsecond * (1 + z)

	#Subtract the continuum from the flux, just use polynomial fit right now 
	counts = flux - continuum
	
	########################################################################
	#FITTING EACH OF THE EMISSION LINES IN TURN
	########################################################################
	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength
	#Can always revert to the old way of doing it by just replacing 
	#the fit_ quantities with what they were before 
	#Or looking in the git commit section and accessing the old file

	#Use np.where to find the indices of data surrounding the gaussian
	index = np.where( wavelength > (H_alpha_shifted - 10) )
	new_wavelength = wavelength[index]
	new_counts = counts[index]
	new_error = error[index]
	new_index = np.where( new_wavelength < (H_alpha_shifted + 10))

	#Select only data for the fit with these indices
	fit_wavelength = new_wavelength[new_index]
	fit_counts = new_counts[new_index]
	fit_error = new_error[new_index]

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
	pars['center'].set(value = H_alpha_shifted)
	pars['center'].set(vary = 'False')

	#Now perform the fit to the data using the set and guessed parameters 
	#And the inverse variance weights form the fits file 
	out  = mod.fit(fit_counts, pars, weights = fit_error, x=fit_wavelength)
	print(out.fit_report(min_correl=0.25))

	#Plot the results and the spectrum to check the fit
	plt.plot(fit_wavelength, out.best_fit, 'r-')
	plt.plot(wavelength, counts)

	#The amplitude parameter is the area under the curve, equivalent to the flux
	amplitude_Ha = out.best_values['amplitude']
	width_Ha = out.best_values['sigma']
	fwhm_Ha = out.best_values['fwhm']
	
	#Store these values inside a results vector for H_a 
	H_a_results = [amplitude, width_Ha, fwhm_Ha]

	#Repeat the process for the other emission lines, for loop? 


	#The return dictionary for this method is a sequence of results vectors
	return {'H_a_results': H_a_results, 'H_B_results': H_B_results, 
	'OIII4959_results': OIII4959_results, 'OIII5007_results': OIII5007_results, }



data=np.genfromtxt('names.txt', dtype=None)
for name in data:
	dict_results = readFits(name)
	flux = dict_results['flux']		
	wavelength = dict_results['wavelength']
	z = dict_results['z']
	continuum = dict_results['continuum']
	error = dict_results['error']

	#Fit the emission lines using the defined methods
	flux_Ha = fitLines(flux, wavelength, z, continuum, error)

	#The given flux is in 10^-17 ergs / s / cm^2
	flux_Ha = flux_Ha * 1E-17 

	#fitHbeta(flux, wavelength, z, continuum, error)
	#fitOIIIfirst(flux, wavelength, z, continuum, error)
	#fitOIIIsecond(flux, wavelength, z, continuum, error)

	#Find the luminosity distance of the object at redshift z 
	#Convert the luminosity distance to cm  
	D_L = toolkit.cosmoDistanceQuad(z, 69.6, 0.286, 0.714)['D_L']

	D_L = D_L * 3.086E26
	

	#Compute H-alpha Luminosity in ergs / s
	Lum_Ha = 4 * np.pi * (D_L**2) * flux_Ha
	print Lum_Ha

	#Compute the star formation rate from the Kennicutt formula 
	SFR = np.log10(Lum_Ha) - 41.27
	print SFR
	print flux_Ha
	plt.show()

	#since the above is a bit rubbish let's try fitting with lmfit 
	#HELLO


