#import the relevant modules
import os, sys, numpy as np, random, matplotlib.mlab as mlab, matplotlib.pyplot as plt, math, matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
from matplotlib.colors import LogNorm
import numpy.polynomial.polynomial as poly

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

def fitGaussian(flux, wavelength, z, continuum, error): 	

#First attempt fitting a gaussian with all the data before masking off most of it
#Will first fit the H-alpha line so define the wavelength of H-alpha (angstroms)
#################################################################
#POSSIBLY THESE SHOULD INVOLVE A VACUUM CORRECTION 
#################################################################

	H_alpha = 6564.614
	#Now apply the redshift formula to find where this will be observed
	H_alpha_shifted = H_alpha * (1 + z)
	#Subtract the continuum from the flux 
	counts = flux - continuum
	#Fit a gaussian to this emission line by optimal scaling of the parameter A 
	#Have to choose a value for the spectral resolution R 
	R = 0.1

	function = (1 / (np.sqrt(np.pi) * R)) * np.exp(-0.5 * ((wavelength - H_alpha_shifted)/R)**2) 	
	A_ml = sum(((counts * function)/error**2)) / sum((function**2)/error**2)
	plt.plot(wavelength, counts, label='flux')
	plt.plot(wavelength, (A_ml * function), label='Gaussian')
	legend()
	plt.show()
	plt.close('all')



data=np.genfromtxt('names.txt', dtype=None)
for name in data:
	dict_results = readFits(name)
	flux = dict_results['flux']		
	wavelength = dict_results['wavelength']
	z = dict_results['z']
	continuum = dict_results['continuum']
	error = dict_results['error']
	fitGaussian(flux, wavelength, z, continuum, error)

