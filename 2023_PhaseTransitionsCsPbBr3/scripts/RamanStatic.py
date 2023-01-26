#! /usr/bin/env python

import os
import numpy as np
from molmod.units import *
from molmod.constants import *
from yaff import *
from molmod.periodic import periodic
from molmod.io.xyz import XYZReader
from molmod import UnitCell
from molmod.io.xyz import XYZFile, XYZWriter
from vasp_born_epsilon import *
from vasp_freq import *

nr_atoms = 20
alpha = np.zeros((nr_atoms,3,3,3))
vasprun = VASPRun('001/1/vasprun.xml')
masses = vasprun.fields['masses']
for i in range(nr_atoms):
    for j in range(3):
        dielectric = np.zeros((2,3,3))
        for k in range(2):
            vasprun_LR = VASPRun_LR('{:03}/{}/vasprun.xml'.format(3*i+j+1,k+1), field_labels=['born','dielectric'])
            dielectric[k,:,:] = vasprun_LR.fields['dielectric']
        alpha[i,j,:,:] = (dielectric[1,:,:]-dielectric[0,:,:])#/masses[i]

frequencies = []
eigenvectors = []
with open('eigenmodes.txt','r') as f:
    counter = 1
    reset = nr_atoms + 3
    eigvec = []
    for line in f:
        reset += 1
        a = line.split()
        if len(a) == 2:
            reset = 1
            frequencies.append(float(a[1]))
        if (reset > 3 and reset < nr_atoms + 4):
            eigvec.append([float(a[2]), float(a[3]), float(a[4])])
        if reset == nr_atoms + 3:
            eigenvectors.append(eigvec)
            eigvec = []

frequencies = np.asarray(frequencies)
eigenvectors = np.asarray(eigenvectors)

print(np.shape(alpha))
print(np.shape(eigenvectors))
alpha_mode = np.einsum('aij,ijkl',eigenvectors,alpha)
print(np.shape(alpha_mode))

amps_iso = np.zeros(len(alpha_mode))
amps_aniso = np.zeros(len(alpha_mode))
for i in range(len(alpha_mode)):
    amps_iso[i] = 1./3.*np.einsum('ii',alpha_mode[i])
    amps_aniso[i] = 1./2.*(3*np.sum(alpha_mode[i]**2)-9*amps_iso[i]**2)

amps_total = 45*amps_iso**2 + 7*amps_aniso

filetxt = open('intensities.txt','w')
for i in range(len(frequencies)):
    intens = str('{:5.20f}'.format(amps_total[i]))
    freq = str('{:8.4f}'.format(frequencies[i]))
    filetxt.write(freq)
    filetxt.write('\t')
    filetxt.write(intens)
    filetxt.write('\n')
filetxt.close()
