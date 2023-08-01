import numpy as np
from molmod.units import *
from molmod.periodic import *
from QEinput import *
import os
import shutil

natoms = 456
force_deriv = np.zeros((3,3,natoms,3))
force_deriv2 = np.zeros((3,3,natoms,3))
force_deriv3 = np.zeros((3,3,natoms,3))
force = np.zeros((5,natoms,3))
#x-axis
for i, element in enumerate(['m200','m100','p000','p100','p200']):
    print i,element
    QE = QE_xml('{}/UiO66.xml'.format(element))
    force[i] = QE.fields['forces']
force_deriv[0,0] = -force[0] + 16*force[1] - 30*force[2] + 16*force[3] - force[4]
force_deriv2[0,0] = force[0] - 2*force[2] + force[4]
force_deriv3[0,0] = force[1] - 2*force[2] + force[3]

#y-axis
for i, element in enumerate(['m020','m010','p000','p010','p020']):
    print i,element
    QE = QE_xml('{}/UiO66.xml'.format(element))
    force[i] = QE.fields['forces']
force_deriv[1,1] = -force[0] + 16*force[1] - 30*force[2] + 16*force[3] - force[4]
force_deriv2[1,1] = force[0] - 2*force[2] + force[4]
force_deriv3[1,1] = force[1] - 2*force[2] + force[3]

#z-axis
for i, element in enumerate(['m002','m001','p000','p001','p002']):
    print i,element
    QE = QE_xml('{}/UiO66.xml'.format(element))
    force[i] = QE.fields['forces']
force_deriv[2,2] = -force[0] + 16*force[1] - 30*force[2] + 16*force[3] - force[4]
force_deriv2[2,2] = force[0] - 2*force[2] + force[4]
force_deriv3[2,2] = force[1] - 2*force[2] + force[3]

#xy-axis
for i, element in enumerate(['m220','m110','p000','p110','p220']):
    print i,element
    QE = QE_xml('{}/UiO66.xml'.format(element))
    force[i] = QE.fields['forces']
force_deriv[0,1] = 1./2.*(-force[0] + 16*force[1] - 30*force[2] + 16*force[3] - force[4]) - 1./2.*force_deriv[0,0] - 1./2.*force_deriv[1,1]
force_deriv[1,0] = force_deriv[0,1]
force_deriv2[0,1] = 1./2.*(force[0] - 2*force[2] + force[4]) - 1./2.*force_deriv2[0,0] - 1./2.*force_deriv2[1,1]
force_deriv2[1,0] = force_deriv2[0,1]
force_deriv3[0,1] = 1./2.*(force[1] - 2*force[2] + force[3]) - 1./2.*force_deriv3[0,0] - 1./2.*force_deriv3[1,1]
force_deriv3[1,0] = force_deriv3[0,1]

#xz-axis
for i, element in enumerate(['m202','m101','p000','p101','p202']):
    print i,element
    QE = QE_xml('{}/UiO66.xml'.format(element))
    force[i] = QE.fields['forces']
force_deriv[0,2] = 1./2.*(-force[0] + 16*force[1] - 30*force[2] + 16*force[3] - force[4]) - 1./2.*force_deriv[0,0] - 1./2.*force_deriv[2,2]
force_deriv[2,0] = force_deriv[0,2]
force_deriv2[0,2] = 1./2.*(force[0] - 2*force[2] + force[4]) - 1./2.*force_deriv2[0,0] - 1./2.*force_deriv2[2,2]
force_deriv2[2,0] = force_deriv2[0,2]
force_deriv3[0,2] = 1./2.*(force[1] - 2*force[2] + force[3]) - 1./2.*force_deriv3[0,0] - 1./2.*force_deriv3[2,2]
force_deriv3[2,0] = force_deriv3[0,2]

#yz-axis
for i, element in enumerate(['m022','m011','p000','p011','p022']):
    print i,element
    QE = QE_xml('{}/UiO66.xml'.format(element))
    force[i] = QE.fields['forces']
force_deriv[1,2] = 1./2.*(-force[0] + 16*force[1] - 30*force[2] + 16*force[3] - force[4]) - 1./2.*force_deriv[1,1] - 1./2.*force_deriv[2,2]
force_deriv[2,1] = force_deriv[1,2]
force_deriv2[1,2] = 1./2.*(force[0] - 2*force[2] + force[4]) - 1./2.*force_deriv2[1,1] - 1./2.*force_deriv2[2,2]
force_deriv2[2,1] = force_deriv2[1,2]
force_deriv3[1,2] = 1./2.*(force[1] - 2*force[2] + force[3]) - 1./2.*force_deriv3[1,1] - 1./2.*force_deriv3[2,2]
force_deriv3[2,1] = force_deriv3[1,2]

frequencies = []
eigenvectors = []
with open('eigenmodes.txt','r') as f:
    counter = 1
    reset = natoms + 3
    eigvec = []
    for line in f:
        reset += 1
        a = line.split()
        if len(a) == 2:
            reset = 1
            frequencies.append(float(a[1]))
        if (reset > 3 and reset < natoms + 4):
            eigvec.append([float(a[2]), float(a[3]), float(a[4])])
        if reset == natoms + 3:
            eigenvectors.append(eigvec)
            eigvec = []

frequencies = np.asarray(frequencies)
eigenvectors = np.asarray(eigenvectors)

alpha_mode = np.einsum('aij,klij',eigenvectors,force_deriv)
alpha_mode2 = np.einsum('aij,klij',eigenvectors,force_deriv2)
alpha_mode3 = np.einsum('aij,klij',eigenvectors,force_deriv3)

amps_iso = np.zeros(len(alpha_mode))
amps_aniso = np.zeros(len(alpha_mode))
for i in range(len(alpha_mode)):
    amps_iso[i] = 1./3.*np.einsum('ii',alpha_mode[i])
    amps_aniso[i] = 1./2.*(3*np.sum(alpha_mode[i]**2)-9*amps_iso[i]**2)
amps_total = 45*amps_iso**2 + 7*amps_aniso

amps_iso2 = np.zeros(len(alpha_mode2))
amps_aniso2 = np.zeros(len(alpha_mode2))
for i in range(len(alpha_mode2)):
    amps_iso2[i] = 1./3.*np.einsum('ii',alpha_mode2[i])
    amps_aniso2[i] = 1./2.*(3*np.sum(alpha_mode2[i]**2)-9*amps_iso2[i]**2)
amps_total2 = 45*amps_iso2**2 + 7*amps_aniso2

amps_iso3 = np.zeros(len(alpha_mode3))
amps_aniso3 = np.zeros(len(alpha_mode3))
for i in range(len(alpha_mode2)):
    amps_iso3[i] = 1./3.*np.einsum('ii',alpha_mode3[i])
    amps_aniso3[i] = 1./2.*(3*np.sum(alpha_mode3[i]**2)-9*amps_iso3[i]**2)
amps_total3 = 45*amps_iso3**2 + 7*amps_aniso3

filetxt = open('intensities.txt','w')
for i in range(len(frequencies)):
    intens = str('{:5.20f}'.format(amps_total[i]))
    freq = str('{:8.4f}'.format(frequencies[i]))
    filetxt.write(freq)
    filetxt.write('\t')
    filetxt.write(intens)
    filetxt.write('\n')
filetxt.close()

filetxt = open('intensities2.txt','w')
for i in range(len(frequencies)):
    intens = str('{:5.20f}'.format(amps_total2[i]))
    freq = str('{:8.4f}'.format(frequencies[i]))
    filetxt.write(freq)
    filetxt.write('\t')
    filetxt.write(intens)
    filetxt.write('\n')
filetxt.close()

filetxt = open('intensities3.txt','w')
for i in range(len(frequencies)):
    intens = str('{:5.20f}'.format(amps_total3[i]))
    freq = str('{:8.4f}'.format(frequencies[i]))
    filetxt.write(freq)
    filetxt.write('\t')
    filetxt.write(intens)
    filetxt.write('\n')
filetxt.close()
