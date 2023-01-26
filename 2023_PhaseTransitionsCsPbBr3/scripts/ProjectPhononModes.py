#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as pt
from molmod.units import *
from molmod.constants import *
from molmod.unit_cells import UnitCell
from molmod.io import dump_chk, load_chk, XYZWriter
from molmod.io.xyz import *
from molmod.periodic import periodic
from yaff import *
from tamkin import NMA, Molecule,ConstrainExt
from tamkin.io import load_molecule_vasp, dump_modes_xyz, dump_modes_molden

#Load static data
chk = load_chk('vasp.chk')
numbers_VASP = chk['numbers']
hessian_VASP = chk['hessian'].reshape(3*len(numbers_VASP),3*len(numbers_VASP))
masses_VASP = chk['masses']
pos_VASP = chk['pos']
gradient = chk['gradient']
energy = chk['ene']
rvecs = chk['rvecs'].flatten()

mol = Molecule(numbers_VASP, pos_VASP, masses_VASP, energy, gradient, hessian_VASP)
nma = NMA(mol)

nmode = nma.modes.shape[1]
freqs = nma.freqs
modes = nma.modes
masses = nma.masses
numbers = nma.numbers
coordinates = nma.coordinates

### convert modes to the right convention
masses3_sqrt1 = np.array(sum([[1/m,1/m,1/m] for m in np.sqrt(masses)],[]))
nmode = modes.shape[1]
modes = modes.copy() # avoid modifying the given modes
for imode in xrange(nmode):
    modes[:,imode] /= np.linalg.norm(modes[:,imode])
    modes[:,imode] *= masses3_sqrt1

#Load dynamic data
xyz = XYZFile('CsPbBr3_gamma_100-nvt.xyz')
masses = np.asarray([periodic[xyz.numbers[i]].mass for i in range(len(xyz.numbers))])
pos = np.reshape(xyz.geometries, (len(xyz.geometries),-1))

#generate projected trajectories
num_cells = len(xyz.geometries[0])/len(numbers)
mode_sc = np.tile(modes,(num_cells,1))
print num_cells, np.shape(modes), np.shape(mode_sc), np.shape(pos)

pos_proj = np.einsum('ij,jk',pos,mode_sc)
print np.shape(pos_proj)

#Define trajectories
step = np.arange(0,len(pos_proj),1)
time = step*2.0*femtosecond
f = h5py.File('VPS_mode.h5', 'w')
f['trajectory/time'] = time
f['trajectory/step'] = step

for i in range(len(modes)):
    f['trajectory/VPS_mode{}'.format(i)] = pos_proj[:,i]
f.close()

#Vibrational spectrum
for i in range(len(modes)):
    f = h5py.File('VPS_mode.h5', 'a')
    spectrum = Spectrum(f, step=5, start=10000, end=50000, path='trajectory/VPS_mode{}'.format(i), bsize=8000, outpath='trajectory/VPS_spectrum_mode{}'.format(i))
    spectrum.compute_offline()
    xunit = lightspeed/centimeter
    spectrum.amps = spectrum.amps*(spectrum.freqs/xunit)**2
    spectrum.freqs = spectrum.freqs/xunit
    np.savetxt('VPS_intensities_mode{}'.format(i), np.c_[spectrum.freqs, spectrum.amps])
