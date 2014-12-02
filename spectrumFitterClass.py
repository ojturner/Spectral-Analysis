#This class houses the methods which are relevant 
#For fiting gaussians to a galactic spectra, 
#fitting template spectra to an observed spectrum 
#and estimating the galactic physical properties 

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
from astropy.io import fits

####################################################################

class spectrumFit(object):
	#Initialiser creates an instance of the spectrumFit object 
	def __init__(self, inFile):
		#The fits file to read the spectral data from
		self.inFile = inFile

####################################################################

######################################################################################
#MODULE: fitPoly
#
#PURPOSE:
#Fit a polynomial to the underlying continuum as a first guess
#No emission line masking in this, as it isn't relevant 
#
#INPUTS:
#
#			flux: The flux values to fit 
#			wavelength: The corresponding wavelength values
#
#
#OUTPUTS: 	continuum: The best fitting polynomial model to the continuum
#						
#
#USAGE: 	continuum = fitPoly(flux, wavelength)
#######################################################################################	

	def fitPoly(self, flux, wavelength):
		#Create the polynomial model from lmFit (from lmfit import PolynomialModel)
		mod = PolynomialModel(6)
		#Have an initial guess at the model parameters 
		pars = mod.guess(flux, x=wavelength)
		#Use the parameters for the full model fit 
		out  = mod.fit(flux, pars, x=wavelength)
		#The output of the model is the fitted continuum
		continuum = out.best_fit
		#return this value
		return continuum

	
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
	#Now define all of the methods associated with the class		
	def readFits(self): 
	
		#Open the fits file, also want to do this using astropy.io.fits instead  
		table = fits.open(self.inFile)
		#Only interested in the data, not the HDU
		data = table[1].data
		#Look for the flux field
		flux = data.field('flux')
		#Wavelength in logarithmic units 
		wavelength = 10**(data.field('loglam'))
		#Inverse variance column
		ivar = data.field('ivar')
		#Call this weights for some reason 
		weights = ivar
		#Can also access the redshift column inside these particular fits files
		redshift_data = table[2].data.field('Z')
		#Use the fitPoly method to find the continuum 
		continuum = self.fitPoly(flux, wavelength)
		#At the moment get a crude estimate of the observed normalised SED for redshift computation
		normalised_observed_flux = flux - continuum
	
		#Return a dictionary of the results 
		return {'flux' : flux, 'wavelength' : wavelength,
		'z' : redshift_data, 'weights': weights, 'norm_flux':normalised_observed_flux}	

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


	def templateRedshift(self, t_wavelength, obs_wavelength, t_flux, flux):

		#Convert to numpy arrays
		t_wavelength = np.array(t_wavelength)
		obs_wavelength = np.array(obs_wavelength)
		t_flux = np.array(t_flux)
		flux = np.array(flux)

		#Create grids of the wavelength and flux values 
		obs_grid = np.array([obs_wavelength, flux])
		t_grid = np.array([t_wavelength, t_flux])

		#Find the resolution of both the template and observed wavelength arrays
		r_template = (t_wavelength[1] - t_wavelength[0]) / len(t_wavelength)
		r_observed = (obs_wavelength[1] - obs_wavelength[0]) / len(obs_wavelength)

		#Find the maximum and minimum wavelength of both arrays 
		min_wavelength = min(min(t_wavelength), min(obs_wavelength))
		max_wavelength = max(max(t_wavelength), max(obs_wavelength))
	

		#Decide which wavelength array has the highest resolution
		#Use this to populate the grid of wavelengths. 
		d_lamba_obs = obs_wavelength[1] - obs_wavelength[0]
		d_lamba_temp = t_wavelength[1] - t_wavelength[0]
		obs_division = (max_wavelength - min_wavelength) / len(obs_wavelength)
		t_division = (max_wavelength - min_wavelength) / len(t_wavelength)


		if r_template > r_observed: 
			wavelength_grid = np.arange(min_wavelength, max_wavelength, t_division)
		else: 
			wavelength_grid = np.arange(min_wavelength, max_wavelength, obs_division)

		#Grid now paritally setup: have to bin up the flux values corresponding to 
		#the range in wavelength spanned by each step in the wavelength grid
		#First create a final wavelength and flux grid for both the template and 
		#the observed spectrum and initially fill the flux values with 0's and the
		#wavelength values with the wavelength grid from above
		flux_zeros = np.zeros(len(wavelength_grid))
		obs_grid_final = np.array([wavelength_grid, flux_zeros])
		t_grid_final = np.array([wavelength_grid, flux_zeros])

		#Now fairly complicated for loop to compute the flux values for both of these grids
		for i in range((len(obs_grid_final[0]) - 1)): 
			#Create an empty array to house the flux average
			flux_avg = []
			#find the indices of the wavelength values in the correct range  
			index_list = np.where(np.logical_and(
				obs_grid[0] > obs_grid_final[0][i], 
				obs_grid[0] < obs_grid_final[0][i + 1]))

			#If the index list is empty, set the flux value to 0
			if len(index_list[0]) == 0:
				obs_grid_final[1][i] = 0.0

			#Otherwise average all of the flux values within the interval	
			else:
			
				#Now find the flux values that these indices correspond to and add to flux avg array
				for entry in index_list:
					flux_avg.append(obs_grid[1][entry])
				#Append the mean of the averaged fluxes to the final grid 
				obs_grid_final[1][i] = np.mean(flux_avg)		

		#Repeat the same process for the template flux grid
		for i in range((len(t_grid_final[0]) - 1)): 
			#Create an empty array to house the flux average
			flux_avg = []
			#find the indices of the wavelength values in the correct range  
			index_list = np.where(np.logical_and(
				t_grid[0] > t_grid_final[0][i], 
				t_grid[0] < t_grid_final[0][i + 1]))

			#If the index list is empty, set the flux value to 0
			if len(index_list[0]) == 0:
				t_grid_final[1][i] = 0.0

			#Otherwise average all of the flux values within the interval	
			else:
			
				#Now find the flux values that these indices correspond to and add to flux avg array
				for entry in index_list:
					flux_avg.append(t_grid[1][entry])
				#Append the mean of the averaged fluxes to the final grid 
				t_grid_final[1][i] = np.mean(flux_avg)	

		obs_grid_final[1][-1] = obs_grid[1][-1]	
		t_grid_final[1][-1] = t_grid[1][-1]	



		print t_grid_final[1], obs_grid_final[1]
		print len(t_grid_final[0]), len(obs_grid_final)

		#Now we have the template and observed fluxes on a common footing 
		#How do we take the cross correlation of these two? 

		if r_template > r_observed: 
			k = np.arange(0, len(t_grid_final[1])*t_division, t_division)
		else: 
			k = np.arange(0, len(obs_grid_final[1])*obs_division, obs_division)


		#Maybe need to reverse the order of the second vector
		ccf = np.correlate(t_grid_final[1], obs_grid_final[1], mode='same')
		plt.close('all')

		plt.plot(k, ccf)
		plt.show()
		print ccf.argmax()

##########################################################################################
#MODULE: fluxError
#
#PURPOSE:
#Monte carlo simulation for computing the error on the fitted flux measurement 
#and also the error on the equivalent width 
#
#INPUTS:
#			counts: photon counts surrounding a particular emission line
#			wavelength: corresponding wavelength values attached to these fluxes
#			error: The errors on the counts 
#			continuum: the stellar continuum vector (counts is assumed already continuum subtracted) 		
#			
#
#OUTPUTS: 	error_dict = dictionary containing the keys flux_error and eq_error 
#
#USAGE: 	redshift = templateRedshift(wavelength, t_flux, flux)
#######################################################################################



	def fluxError(self, counts, wavelength, error, continuum):
		flux_vector = []
		E_W_vector = []
		cont_avg = np.mean(continuum)
		#plt.close('all')

		for i in range(100):
			#plt.errorbar(wavelength, counts, yerr=error)
			new_counts=[]
			j = 0
			for point in counts:
				new_counts.append(np.random.normal(point, error[j]))
				j = j + 1
			new_counts = np.array(new_counts)
		#So for each N in 1000 a new counts vector is generated randomly
		#Take this data against the wavelength values and fit a gaussian 
		#each time to compute the flux. Append this to a vector and then 
		#find the standard deviation to get the flux error for that emission line
		#Note this is to be encorporated in the fitLines module so that each of the emission 
		#lines is fit in turn. Next step here is to construct the model with lmfit, 
		#guess the initial parameters and then fit the gaussian and take 
		#out.best_values('amplitude') as the flux and store in flux_vector	

		#Now use the lmfit package to perform gaussian fits to the data	
		#Construct the gaussian model
			mod = GaussianModel()

	#Take an initial guess at what the model parameters are 
	#In this case the gaussian model has three parameters, 
	#Which are amplitude, center and sigma
			pars = mod.guess(new_counts, x=wavelength)

	#We know from the redshift what the center of the gaussian is, set this
	#And choose the option not to vary this parameter 
	#Leave the guessed values of the other parameters
			pars['center'].set(value = np.mean(wavelength))


	#Now perform the fit to the data using the set and guessed parameters 
	#And the inverse variance weights form the fits file 
			out  = mod.fit(new_counts, pars, x=wavelength)
			flux = out.best_values['amplitude']
			E_W = out.best_values['amplitude'] / cont_avg
			flux_vector.append(flux)
			E_W_vector.append(E_W)
		#plt.scatter(wavelength, new_counts)
		#plt.plot(wavelength, out.best_fit)
		

		print 'Hello', flux_vector
	#Now return the standard deviation of the flux_vector as the flux error 
		return {'flux_error' : np.std(flux_vector), 'E_W_error' : np.std(E_W_vector)}

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


#Will try fitLines using a composite model i.e. multi gaussian fit

	def fitLines(self, flux, wavelength, z, weights): 	

		#Convert all into numpy arrays 
		flux = np.array(flux)
		wavelength = np.array(wavelength)
		z = np.array(z)
		weights = np.array(weights)
		error = np.sqrt(1 / weights)

		#Fit a polynomial to the continuum background emission of the galaxy
		#This is the crude way to do it 
		continuum_poly = self.fitPoly(flux, wavelength)

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

			if (i + 5) < len(masked_flux):
				continuum[i] = ma.mean(masked_flux[i:i+5])
				if np.isnan(continuum[i]):
					continuum[i] = continuum[i - 1]
			else:
				continuum[i] = ma.mean(masked_flux[i-5:i])
				if np.isnan(continuum[i]):
					continuum[i] = continuum[i - 1]

		

		#Subtract the continuum from the flux, just use polynomial fit right now 
		counts = flux - continuum_poly

		#Construct a dictionary housing these shifted emission line values 
		#Note that values for the OII doublet are not present
		line_dict = {'H_beta' : H_beta_shifted, 'OIII4959' : OIII4959_shifted, 
		'OIII5007' : OIII5007_shifted, 'H_alpha' : H_alpha_shifted, 'NII6585' : NII6585_shifted, 
		'SII6718' : SII6718_shifted, 'SII6732' : SII6732_shifted}

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
			new_index = np.where(np.logical_and(wavelength > (line_dict[key] - 10) ,
												wavelength < (line_dict[key] + 10)))  

		#Select only data for the fit with these indices
			fit_wavelength = wavelength[new_index]
			fit_counts = counts[new_index]
			fit_weights = weights[new_index]
			fit_continuum = continuum[new_index]
			fit_error = error[new_index]



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
		
		#Return the error on the flux 
			error_dict = self.fluxError(fit_counts, fit_wavelength, fit_error, continuum_poly)

		#Compute the equivalent width
			con_avg = np.mean(continuum_poly)
			E_w = out.best_values['amplitude'] / con_avg

		#The amplitude parameter is the area under the curve, equivalent to the flux
			results_dict[key] = [out.best_values['amplitude'], error_dict['flux_error'], out.best_values['sigma'], 
			2.3548200*out.best_values['sigma'], E_w, error_dict['E_W_error']]

		plt.show()	
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

	def galPhys(self, z, flux_Ha, flux_NeII6585, flux_OIII5007, flux_Hb):

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

	
######################################################################################
#Create a second class for all methods related to the template spectrum
######################################################################################

class templateSpec(object):
	#Initialiser creates an instance of the templateSpec object 
	def __init__(self, inFile):
		#The ascii file to read the spectral data from
		self.inFile = inFile

#######################################################################################

	##########################################################################################
	#MODULE: normaliseTemplate
	#
	#PURPOSE:
	#Take an ascii file with template galaxy wavelength and flux values, computed from a galaxy SED code 
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

	def normaliseTemplate(self):

		#Read in the file and assign the flux and wavelength vectors
		template = np.loadtxt(self.inFile)
		wavelength = template[:,0]
		flux = template[:,1]


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
				continuum[i] = ma.median(masked_flux[i:i+10])
				if np.isnan(continuum[i]):
					continuum[i] = continuum[i - 1]
			else:
				continuum[i] = ma.median(masked_flux[i-10:i])
				if np.isnan(continuum[i]):
					continuum[i] = continuum[i - 1]

		#Which works, (so long as I'm careful with masked values)
		#Now divide by the continuum and plot the final template

		normalised_flux = flux - continuum

		#Show off the plots
		plt.plot(wavelength, continuum, color='purple', linewidth=2, label='continuum')
		plt.plot(wavelength, normalised_flux, color='black', linewidth=2, label='normalised')
		plt.plot(wavelength, flux, color='green', label='template_spectrum')
		plt.plot(wavelength, masked_flux, color='red', label='masked')
		plt.show()
		

		return {'norm_flux' : normalised_flux, 'wavelength' : wavelength}










