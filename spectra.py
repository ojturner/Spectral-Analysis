#import the relevant modules
import os, sys, numpy as np, random, matplotlib.mlab as mlab, matplotlib.pyplot as plt, math, matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
from matplotlib.colors import LogNorm
import numpy.polynomial.polynomial as poly
import lmfit
from lmfit.models import GaussianModel, ExponentialModel, LorentzianModel, VoigtModel

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
	error = np.sqrt(1/ivar)
	y_av = toolkit.movingaverage(flux, 50)
	coefs = poly.polyfit(wavelength, flux, 8, w=ivar)
	ffit = poly.polyval(wavelength, coefs)

		#Choose to plot the results 
	#plt.plot(wavelength, flux, label='flux')
	#plt.plot(wavelength, y_av, label='boxcar')
	#plt.plot(wavelength, ffit , label='fit')
	#legend()
	#plt.show()
	#plt.close('all')

	redshift_data = table[2].data.field('Z')
	return {'flux' : flux, 'wavelength' : wavelength, 'z' : redshift_data, 'continuum': ffit, 'error': error}

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


	#Fit a gaussian to this emission line by optimal scaling of the parameter A 
	#Have to choose a value for the spectral resolution R 
	

	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength
	#Can always revert to the old way of doing it by just replacing 
	#the fit_ quantities with what they were before 
	#Or looking in the git commit section and accessing the old file

	index = np.where( wavelength > (H_alpha_shifted - 10) )
	new_wavelength = wavelength[index]
	new_counts = counts[index]
	new_error = error[index]
	new_index = np.where( new_wavelength < (H_alpha_shifted + 10))
	fit_wavelength = new_wavelength[new_index]
	fit_counts = new_counts[new_index]
	fit_error = new_error[new_index]



	#gauss1  = GaussianModel(prefix='g1_')

	#pars = gauss1.make_params()
	#pars.update( gauss1.make_params())

	#pars['g1_center'].set(H_alpha_shifted)
	#pars['g1_sigma'].set(1)
	#pars['g1_amplitude'].set(80, min=0, max=300)

	#gauss2  = GaussianModel(prefix='g2_')
	#pars.update(gauss2.make_params())

	#pars['g2_center'].set(H_beta_shifted)
	#pars['g2_sigma'].set(1)
	#pars['g2_amplitude'].set(10)

	#mod = gauss1 + gauss2 

	#init = mod.eval(pars, x=wavelength)
	
	#plt.plot(wavelength, init, 'k--')

	#out = mod.fit(counts, pars, x=wavelength)

	#print(out.fit_report())
	#print pars

	

	mod = GaussianModel()
	pars = mod.guess(fit_counts, x=fit_wavelength)
	pars['center'].set(H_alpha_shifted)
	#pars['amplitude'].set(400, min=0, max=1000)
	out  = mod.fit(fit_counts, pars, x=fit_wavelength)
	print(out.fit_report(min_correl=0.25))
	plt.plot(fit_wavelength, out.best_fit, 'r-')
	plt.plot(wavelength, counts)
	#plt.show()
	amplitude = out.best_values['amplitude']
	width = out.best_values['sigma']
	


	#function = (1 / (np.sqrt(np.pi) * R)) * np.exp(-0.5 * ((fit_wavelength - H_alpha_shifted)/R)**2) 	
	#A_ml = sum(((fit_counts * function)/fit_error**2)) / sum((function**2)/fit_error**2)

	#indx = np.where((A_ml * function) > 1)
	#flux_Ha = (A_ml * function)[indx]

	#plt.plot(wavelength, counts, label='flux')
	#plt.plot(fit_wavelength, (A_ml * function), color='red', linewidth = 2, label='H-alpha')
	#legend()

	return amplitude

def fitHbeta(flux, wavelength, z, continuum, error): 	

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
	#Now apply the redshift formula to find where this will be observed
	H_beta_shifted = H_beta * (1 + z)
	#Subtract the continuum from the flux 
	counts = flux - continuum
	#Fit a gaussian to this emission line by optimal scaling of the parameter A 
	#Have to choose a value for the spectral resolution R 
	R = 0.1

	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength
	#Can always revert to the old way of doing it by just replacing 
	#the fit_ quantities with what they were before 
	#Or looking in the git commit section and accessing the old file

	index = np.where( wavelength > (H_beta_shifted - 20) )
	new_wavelength = wavelength[index]
	new_counts = counts[index]
	new_error = error[index]
	new_index = np.where( new_wavelength < (H_beta_shifted + 20))
	fit_wavelength = new_wavelength[new_index]
	fit_counts = new_counts[new_index]
	fit_error = new_error[new_index]



	function = (1 / (np.sqrt(np.pi) * R)) * np.exp(-0.5 * ((fit_wavelength - H_beta_shifted)/R)**2) 	
	A_ml = sum(((fit_counts * function)/fit_error**2)) / sum((function**2)/fit_error**2)
	plt.plot(fit_wavelength, (A_ml * function), color='green', linewidth = 2, label='H-beta')
	legend()

def fitOIIIfirst(flux, wavelength, z, continuum, error): 	

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


	OIIIfirst = 4960.295
	#Now apply the redshift formula to find where this will be observed
	OIIIfirst_shifted = OIIIfirst * (1 + z)
	#Subtract the continuum from the flux 
	counts = flux - continuum
	#Fit a gaussian to this emission line by optimal scaling of the parameter A 
	#Have to choose a value for the spectral resolution R 
	R = 0.1

	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength
	#Can always revert to the old way of doing it by just replacing 
	#the fit_ quantities with what they were before 
	#Or looking in the git commit section and accessing the old file

	index = np.where( wavelength > (OIIIfirst_shifted - 20) )
	new_wavelength = wavelength[index]
	new_counts = counts[index]
	new_error = error[index]
	new_index = np.where( new_wavelength < (OIIIfirst_shifted + 20))
	fit_wavelength = new_wavelength[new_index]
	fit_counts = new_counts[new_index]
	fit_error = new_error[new_index]



	function = (1 / (np.sqrt(np.pi) * R)) * np.exp(-0.5 * ((fit_wavelength - OIIIfirst_shifted)/R)**2) 	
	A_ml = sum(((fit_counts * function)/fit_error**2)) / sum((function**2)/fit_error**2)
	plt.plot(fit_wavelength, (A_ml * function), color='orange', linewidth = 2, label='O[III]4960')
	legend()

def fitOIIIsecond(flux, wavelength, z, continuum, error): 	

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


	OIIIsecond = 5008.239
	#Now apply the redshift formula to find where this will be observed
	OIIIsecond_shifted = OIIIsecond * (1 + z)
	#Subtract the continuum from the flux 
	counts = flux - continuum
	#Fit a gaussian to this emission line by optimal scaling of the parameter A 
	#Have to choose a value for the spectral resolution R 
	R = 0.1

	#We don't want to include all the data in the gaussian fit 
	#Look for the indices of the points closes to the wavelength value
	#The appropriate range is stored in fit_wavelength
	#Can always revert to the old way of doing it by just replacing 
	#the fit_ quantities with what they were before 
	#Or looking in the git commit section and accessing the old file

	index = np.where( wavelength > (OIIIsecond_shifted - 20) )
	new_wavelength = wavelength[index]
	new_counts = counts[index]
	new_error = error[index]
	new_index = np.where( new_wavelength < (OIIIsecond_shifted + 20))
	fit_wavelength = new_wavelength[new_index]
	fit_counts = new_counts[new_index]
	fit_error = new_error[new_index]



	function = (1 / (np.sqrt(np.pi) * R)) * np.exp(-0.5 * ((fit_wavelength - OIIIsecond_shifted)/R)**2) 	
	A_ml = sum(((fit_counts * function)/fit_error**2)) / sum((function**2)/fit_error**2)
	plt.plot(fit_wavelength, (A_ml * function), color='purple', linewidth = 2, label='O[III]5008.239')
	legend()
	
	



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
	D_L = toolkit.cosmoDistanceQuad(z, 70, 0.28, 0.72)['D_L']

	D_L = D_L * 3.086E26
	print D_L

	#Compute H-alpha Luminosity in ergs / s
	Lum_Ha = 4 * np.pi * (D_L**2) * flux_Ha

	#Compute the star formation rate from the Kennicutt formula 
	SFR = np.log10(Lum_Ha) - 41.27
	print SFR
	print flux_Ha
	#plt.show()

	#since the above is a bit rubbish let's try fitting with lmfit 
	#


