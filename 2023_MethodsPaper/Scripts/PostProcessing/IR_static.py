#!/usr/bin/env python

#Calculation of static IR intensities

from tamkin import NMA, Molecule,ConstrainExt
from molmod.units import centimeter, electronvolt, angstrom, amu
from molmod.constants import lightspeed, planck
from molmod.unit_cells import UnitCell
from molmod import angstrom, lightspeed, centimeter
from molmod.io import XYZWriter as MMXYZWriter
from molmod.periodic import periodic
import numpy as np
from yaff import *
from molmod.io import dump_chk, load_chk
from tamkin.io import load_molecule_vasp, dump_modes_xyz, dump_modes_molden
import matplotlib.pyplot as pt

def lorentz(frequency, x, sigma):
    return 1.0/(np.pi*sigma)*(sigma**2/((x-frequency)**2+sigma**2))  

#read chk file
chk = load_chk('../../Hessian/vasp.chk')
numbers = chk['numbers']
hessian = chk['hessian'].reshape(3*len(numbers),3*len(numbers))
masses = chk['masses']
pos = chk['pos']
gradient = chk['gradient']
energy = chk['ene']

#compute normal modes
mol = Molecule(numbers, pos, masses, energy, gradient, hessian)
nma = NMA(mol)

nmode = nma.modes.shape[1]
freqs = nma.freqs/lightspeed*centimeter
modes = nma.modes.copy()

#read Born effective charge matrix
born = []
with open('born.txt','r') as g:
    counter = 1
    reset = 5
    bornm = []
    for line in g:
        reset += 1
        a = line.split()
        if len(a) == 2:
            reset = 1
        if (reset > 1 and reset < 5):
            bornm.append([float(a[1]), float(a[2]), float(a[3])]) 
        if reset == 4:
            born.append(bornm) 
            bornm = []  
born = np.asarray(born)

#convert to mass-weighted modes
masses3_sqrt1 = np.array(sum([[1/m,1/m,1/m] for m in np.sqrt(masses)],[]))
for imode in xrange(nmode):
    modes[:,imode] /= np.linalg.norm(modes[:,imode])
    modes[:,imode] *= masses3_sqrt1

#calculate intensities
intensities = np.zeros(len(modes))
for i in range(len(modes)):
    mode = np.round(modes[:,i].reshape(len(numbers),3),decimals=6)
    intensities[i] = np.sum(np.einsum('ijk,ik',born,mode)**2)

filetxt = open('intensities.txt','w')
for i in xrange(len(freqs)):
    intens = str('{:5.15f}'.format(intensities[i]))
    freq = str('{:8.4f}'.format(freqs[i]))
    filetxt.write(freq)
    filetxt.write('\t')
    filetxt.write(intens)
    filetxt.write('\n')
filetxt.close()

#plot spectrum
freq = np.linspace(0,4000,10000)
intens = np.zeros(len(freq))
for i in xrange(len(freqs)):
    intens += intensities[i]*lorentz(freqs[i], freq, 5)

pt.figure(figsize=(16,9))
pt.plot(freq,intens/np.max(intens),label='lp',color='blue')
#pt.plot(freq,(1.0-np.exp(-freq*lightspeed/centimeter/boltzmann/300.0))*intens/np.max(intens),label='lp factor',color='red')
pt.xlim(0,4000)
pt.ylim(-0.1,1.1)
pt.ylabel('Absorbance',fontsize=20)
pt.xlabel(r'Wavenumber (cm$^{-1}$)',fontsize=20)
pt.legend()
pt.yticks([])
pt.savefig('UiO66_4brick_IR.pdf', transparent='true',bbox_inches='tight')
pt.show()


