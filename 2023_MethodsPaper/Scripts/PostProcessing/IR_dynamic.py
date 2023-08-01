#! /usr/bin/env python

#Calculation of dynamic IR spectra

import h5py
import numpy as np
import matplotlib.pyplot as pt
from molmod.units import *
from molmod.constants import *
from yaff import *
from molmod.periodic import periodic
from yaff.pes.ext import Cell

f1 = np.genfromtxt('../dipole/E0/dipole.txt')
data = f1[:,0:3]
step = np.arange(0,len(data),1)
time = step*2.0*femtosecond

#Assure continuity of dipole moments
cell = np.genfromtxt('../dipole/cell.txt')

for i in xrange(len(data)-1):
    value = data[i+1,:] - data[i,:] 
    ref = value.copy()
    rvecs = Cell(np.array([cell[:,0],cell[:,1],cell[:,2]]))
    rvecs.mic(value)
    disp = value-ref
    data[i+1,:] += disp 

f = h5py.File('moments.h5', 'w')
f['trajectory/moments'] = data
f['trajectory/time'] = time
f['trajectory/step'] = step
f.close()

#Vibrational spectrum

f = h5py.File('moments.h5', 'a')
spectrum = Spectrum(f,path='trajectory/moments', start=0, end=10000, bsize=10000, key='ir', outpath='trajectory/IR_spectrum')
spectrum.compute_offline()
xunit = lightspeed/centimeter
spectrum.amps = spectrum.amps*(spectrum.freqs/xunit)**2
spectrum.freqs = spectrum.freqs/xunit
np.savetxt('IR_intensities', np.c_[spectrum.freqs, spectrum.amps])
