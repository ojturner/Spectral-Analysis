#Test file which uses the spectrumFit class
import numpy as np
from spectrumFitterClass import spectrumFit
from spectrumFitterClass import templateSpec



#Create an instance of the template object 
template = templateSpec('K20_late_composite_original.dat')

#Manually read in the properties from one of the template spectra
template_dictionary = template.normaliseTemplate()
n_temp_flux = template_dictionary['norm_flux']
temp_wavelength = template_dictionary['wavelength']


#Read the names of the fits files in 
data=np.genfromtxt('names.txt', dtype=None)

#Loop round all of these and create an instance
#of spectrumFit for each name in the file
for name in data:

	#Initialise the spectrumFit object
	spec = spectrumFit(name)
	#Read in the fits file and save the observed properties
	spec_prop = spec.readFits()
	#Fit the emission lines using the defined method, plot the results
	fit_results = spec.fitLines(spec_prop['flux'], spec_prop['wavelength'], \
 		spec_prop['z'], spec_prop['weights'])
	#print fit_results
<<<<<<< HEAD
	print fit_results	
=======
	print fit_results
>>>>>>> compositeModel

	#Now use the galPhys method to compute the physical properties 
	props = spec.galPhys(spec_prop['z'], fit_results['H_alpha'][0] * 1E-17, \
		fit_results['H_beta'][0] * 1E-17, 	fit_results['NII6585'][0] * 1E-17, \
		fit_results['OIII5007'][0] * 1E-17)
	print props

	#Attempt to compute redshift with the templateRedshift method
<<<<<<< HEAD
	spec.templateRedshift(temp_wavelength, spec_prop['wavelength'], \
		n_temp_flux, spec_prop['norm_flux'])
	print spec_prop['z']
=======
	#spec.zFromTemp(temp_wavelength, spec_prop['wavelength'], \
		#n_temp_flux, spec_prop['norm_flux'])
	#print spec_prop['z']
>>>>>>> compositeModel





