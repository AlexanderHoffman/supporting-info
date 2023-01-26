#!/usr/bin/env python

from tamkin import NMA, Molecule,ConstrainExt
from molmod.units import centimeter, electronvolt, angstrom, amu
from molmod.constants import lightspeed, planck
from molmod.unit_cells import UnitCell
from molmod import angstrom, lightspeed, centimeter
from molmod.io import XYZWriter
from molmod.periodic import periodic
import numpy as np
from yaff import *
from molmod.io import dump_chk, load_chk
from tamkin.io import load_molecule_vasp, dump_modes_xyz, dump_modes_molden

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
dump_modes_molden('vasp.log',nma)

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


txt_freq = """\
  Atom  AN          X          Y          Z"""

with open('eigenmodes.txt', 'w') as f:
    for i in xrange(nmode):
        print >> f, 'Frequency', '%10.4f' % (freqs[i]/lightspeed*centimeter)
        print >> f
        print >> f, txt_freq
        
        assert modes.shape[0] % 3 == 0
        natom = modes.shape[0]/3
        
        for iatom in range(natom):
            print >> f, '%6d %3d' % (iatom +1, numbers[iatom]),
            print >> f, '%12.6f %10.6f %10.6f' % (modes[3*iatom, i], modes[3*iatom+1,i], modes[3*iatom+2, i]),
            print >> f
        
        print >> f
        

