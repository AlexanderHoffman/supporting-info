#! /usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as pt
import scipy
from scipy.optimize import curve_fit
from molmod.units import *
from molmod.constants import *

from scipy.optimize import curve_fit

def lorentz(frequency, x, ampl, sigma):
    return ampl/(np.pi*sigma)*(sigma**2/((x-frequency)**2+sigma**2)) 
    
def prefactor(frequency,k_in,T):
    return (k_in-frequency)**4/frequency/(1.0-np.exp(-frequency*lightspeed/centimeter/boltzmann/T)) 

Temp = [50,100,300]

freq_unit = lightspeed/centimeter
freq = np.linspace(0,300,10000)
data_gamma = np.zeros((60,10000,2))
param_gamma = np.zeros((60,len(Temp),3))

for i, value in enumerate(Temp):
    for j in range(3,60):
        h5 = h5py.File('{}/PhononModes/VPS_gamma.h5'.format(value),mode='r')
        freqs = h5['trajectory/VPS_spectrum_mode{}/freqs'.format(j)][:]/freq_unit
        amps = h5['trajectory/VPS_spectrum_mode{}/amps'.format(j)]*freqs**2
        
        data_gamma[j,:,0] = freq
        intens = np.zeros(len(freq))
        for k in range(len(freqs)):
            intens += amps[k]*lorentz(freqs[k], freq, 1.0, 1.0)
        data_gamma[j,:,1] = intens
        param_gamma[j,i,:] = scipy.optimize.curve_fit(lorentz,data_gamma[j,:,0],data_gamma[j,:,1],[100,1,10])[0]

raman_input_gamma = np.genfromtxt('static_raman_intensities.txt') 
freq_raman_gamma = raman_input_gamma[:,0]       
intens_gamma = raman_input_gamma[:,1]/np.max(raman_input_gamma[:,1])

