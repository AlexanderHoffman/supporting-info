#! /usr/bin/env python

#Generation of POSCARs for parallel calculations of dielectric tensors (necessary for Raman tensor)

import os
import numpy as np
from molmod.units import *
from molmod.constants import *
from yaff import *
from molmod.periodic import periodic
from molmod.io.xyz import XYZReader
from molmod import UnitCell
from molmod.io.xyz import XYZFile, XYZWriter

#Read POSCAR
f = open('POSCAR', 'r')
title = f.readline()
scale = float(f.readline())
rvecs = np.zeros([3,3], float)
for i in range(3):
    line = f.readline()
    rvecs[i,:] = np.array([float(word)*scale*angstrom for word in line.split()])
symbols = f.readline().split()
ndups = [int(num) for num in f.readline().split()]
natoms = sum(ndups)
numbers = []
for symbol, ndup in zip(symbols, ndups):
    for i in range(ndup):
        numbers.append(periodic[symbol].number)
numbers = np.array(numbers)
title2 = f.readline().rstrip().lstrip()
if title2.lower() in ['direct', 'cartesian']:
    mode = title2
else:
    mode = f.readline().rstrip().lstrip()
assert mode.lower() in ['direct', 'cartesian'], 'Error reading POSCAR, no Coordinate mode (direct or cartesian) found.'
coords = np.zeros([natoms, 3], float)
for i in range(natoms):
    line = f.readline()
    if mode.lower() == 'direct':
        frac = np.array([float(s) for s in line.split()[:3]])
        coord = np.dot(rvecs.T, frac)
    elif mode.lower() == 'cartesian':
        coord = np.array([float(s)*scale*angstrom for s in line.split()[:3]])
    else:
        raise ValueError('Invalid Coordinate mode specifier, recieved %s' %mode)
    coords[i,:] = coord.copy()

#Make FD POSCARs
for i in range(len(numbers)):
    for j in range(3):
        os.system('mkdir {:03}'.format(3*i+j+1))
        for k in range(2):
            os.system('mkdir {:03}/{}'.format(3*i+j+1,k+1))
            os.system('cp KPOINTS INCAR POTCAR {:03}/{}'.format(3*i+j+1,k+1))
            new_coords = coords.copy()
            new_coords[i,j] += 2*(k-1./2.)*0.01*angstrom

            f = open('{:03}/{}/POSCAR'.format(3*i+j+1,k+1), 'w')
            f.write('atom {}, axis {}, direction {}\n'.format(i+1,j+1,k+1))
            f.write('1.0\n')
            for rvec in rvecs:
                f.write('%20.10f %20.10f %20.10f\n' %(
                    rvec[0]/angstrom, rvec[1]/angstrom, rvec[2]/angstrom
                ))
            f.write(''.join(['%5s' %(symbol) for symbol in symbols])+'\n')
            f.write(''.join(['%5i' %(ndup) for ndup in ndups])+'\n')
            f.write('Cartesian\n')
            for coord in new_coords:
                f.write('%15.9f %15.9f %15.9f\n' %(
                    coord[0]/angstrom, coord[1]/angstrom, coord[2]/angstrom
                ))
            f.close()

