#!/usr/bin/env python

#Extract QHA properties

import os
import numpy as np
import matplotlib.pyplot as pt
from molmod import *
from molmod.periodic import *
from molmod.io import *
from yaff import *
from tamkin import NMA, Molecule

def energy(U,freqs,T):
    E = np.zeros(len(freqs))
    for i in range(len(freqs)):
        E[i] = np.sum(np.pi*freqs[i]*(1+2*np.exp(-np.pi*freqs[i]/boltzmann/T)/(np.exp(np.pi*freqs[i]/boltzmann/T)-np.exp(-np.pi*freqs[i]/boltzmann/T))))
    return E+U

def free_energy(U,freqs,T):
    F = np.zeros(len(freqs))
    for i in range(len(freqs)):
        F[i] = np.sum(np.pi*freqs[i] + boltzmann*T*np.log(1-np.exp(-2*np.pi*freqs[i]/boltzmann/T)))
    return F+U

def entropy(U,freqs,T):
    return (energy(U,freqs,T)-free_energy(U,freqs,T))/T

def heat_capacity(freqs,T):
    return np.sum(boltzmann*(2*np.pi*freqs/boltzmann/T)**2*np.exp(-2*np.pi*freqs/boltzmann/T)/(1-np.exp(-2*np.pi*freqs/boltzmann/T))**2)

def heat_capacity_vol(freqs,T):
    Cv = np.zeros(len(freqs))
    for i in xrange(len(freqs)):
        Cv[i] = np.sum(boltzmann*(2*np.pi*freqs[i]/boltzmann/T)**2*np.exp(-2*np.pi*freqs[i]/boltzmann/T)/(1-np.exp(-2*np.pi*freqs[i]/boltzmann/T))**2)
    return Cv

def pressure(vol,fit_F,degree):
    
    P = np.zeros(len(vol))
    for i in range(degree):
        P += -(degree-i)*fit_F[i]*vol**(degree-1-i)
    
    return P

def bulk_modulus(vol,coeff_F,degree):
    
    K = 0
    for i in range(degree-1):
        K += (degree-i)*(degree-i-1)*coeff_F[i]*vol**(degree-1-i)
    
    return K

def thermal_expansion(Temp,coeff_vol,degree):

    nom = np.zeros(len(Temp))
    denom = np.zeros(len(Temp))
    for i in range(degree):
        nom += (degree-i)*coeff_vol[i]*Temp**(degree-1-i)
        denom += coeff_vol[i]*Temp**(degree-i)
    denom += coeff_vol[degree]
    
    return nom/denom

def plot_free_energy(volumes,energies,freqs,degree=5,Temp=[1,100,200,300,400,500]):

    pt.figure(figsize=(12,9))
    cmap = pt.get_cmap('hot')
    colors = [cmap(i) for i in np.linspace(0,1,100)] 
    reference = 0          
    #process data
    for count,T in enumerate(Temp):
        ene = energy(energies,freqs,T)
        free = free_energy(energies,freqs,T)
        ent = entropy(energies,freqs,T)
        free_coeff = np.polyfit(volumes,free,degree)
        vol = np.linspace(np.min(volumes),np.max(volumes),10000)

        fit = np.zeros(len(vol))
        for i in range(degree+1):
            fit += free_coeff[i]*vol**(degree-i)

        #of via poly1d
        #fit2 = np.poly1d(free_coeff)
        #fit_free = fit2(volumes)
        
        if count==0:
            reference = np.min(fit)

        print np.argmin(fit), vol[np.argmin(fit)], (fit[np.argmin(fit)]-reference)/kjmol

        pt.plot(volumes/angstrom**3,(free-reference)/kjmol, 'o', markersize=8, color=colors[10*(count+1)])
        pt.plot(vol/angstrom**3,(fit-reference)/kjmol, color=colors[10*(count+1)], label='{}'.format(T))
        pt.plot(vol[np.argmin(fit)]/angstrom**3,(fit[np.argmin(fit)]-reference)/kjmol, 'o', markersize=10, color='black')

    pt.xlim(np.min(volumes)/angstrom**3,np.max(volumes)/angstrom**3)
    pt.ylim(-2300,300)
    pt.xlabel('Volume ($\mathrm{\AA}^{3}$)')
    pt.ylabel('Free energy (kJ/mol)')
    pt.legend(ncol=6,loc=9,fontsize=16)
    pt.savefig('Free_energy_UiO66_4brick.pdf',bbox_inches='tight',transparent=True)
    pt.show()

def plot_properties(volumes,energies,freqs,masses,degree=5,Temp=[1,100,200,300,400,500]):

    eq_vol = np.zeros(len(Temp))
    bulk_mod = np.zeros(len(Temp))
    Cv = np.zeros(len(Temp))
    freqs_coeff = np.polyfit(volumes,freqs,degree)

    for count,T in enumerate(Temp):
        free = free_energy(energies,freqs,T)
        free_coeff = np.polyfit(volumes,free,degree)
        vol = np.linspace(np.min(volumes),np.max(volumes),10000)
 
        fit_F = np.zeros(len(vol))
        for i in range(degree+1):
            fit_F += free_coeff[i]*vol**(degree-i)

        index_Fmin = np.argmin(fit_F)
        eq_vol[count] = vol[index_Fmin] 
        bulk_mod[count] = bulk_modulus(vol[index_Fmin],free_coeff,degree)
        eq_freqs = np.zeros(len(freqs[0,:]))
        for i in range(degree+1):
            eq_freqs += freqs_coeff[i]*vol[index_Fmin]**(degree-i) 
        Cv[count] = heat_capacity(eq_freqs,T)
    
    coeff_eq_vol = np.polyfit(Temp,eq_vol,degree) 
    alpha = thermal_expansion(Temp,coeff_eq_vol,degree)
    Cp = Cv + alpha**2*bulk_mod*eq_vol*Temp
    
    #Conversion mol^-1 to g^-1
    molar_mass = 24*periodic[40].mass+192*periodic[6].mass+112*periodic[1].mass+128*periodic[8].mass
    molar_mass *= mol/gram
    
    print Temp[12]
    print bulk_mod[12]/(pascal*10**9), alpha[12]*10**6, Cp[12]/joule*mol/molar_mass

    pt.figure(figsize=(16,9))
    ax1 = pt.subplot(2,2,1)
    ax2 = pt.subplot(2,2,2)
    ax3 = pt.subplot(2,2,3)
    ax4 = pt.subplot(2,2,4)
    ax1.plot(Temp,eq_vol/angstrom**3,color='#d7191c')
    ax2.plot(Temp,alpha*10**6,color='#fdae61')
    ax3.plot(Temp,bulk_mod/(pascal*10**9),color='#abdda4')
    #ax4.plot(Temp,Cp/joule/(np.sum(masses)/gram),color='#2b83ba')
    #ax4.plot(Temp,Cv/joule/(np.sum(masses)/gram),color='black')
    ax4.plot(Temp,Cp/joule*mol/molar_mass,color='#2b83ba')
    #ax4.plot(Temp,Cv/joule*mol,color='black')
    #ax4.hlines(3*75*boltzmann/joule*mol,0,2000,linestyle=(0, (5, 3)),linewidth=1.5)
    ax1.set_xlim(0,500)
    #ax1.set_ylim(-33,180)
    ax1.set_ylabel(r'Volume ($\mathrm{\AA}^{3}$)')
    ax2.set_xlim(0,500)
    #ax2.set_ylim(-33,180)
    ax2.set_ylabel(r'$\mathrm{\alpha}_V$ (10$^{-6}$K$^{-1}$)')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax3.set_xlim(0,500)
    #ax3.set_ylim(-33,180)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel(r'Bulk modulus (GPa)')
    ax4.set_xlim(0,500)
    #ax4.set_ylim(-33,180)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel(r'C$_P$ (Jg$^{-1}$K$^{-1}$)')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    pt.savefig('UiO66_4brick_properties.pdf',transparent=True,bbox_inches='tight')
    pt.show()   
        

#load optimal structure
chk_opt = load_chk('../Hessian/vasp.chk')
numbers = chk_opt['numbers']
masses = chk_opt['masses']
pos = chk_opt['pos']
gradient = chk_opt['gradient']
hessian_opt = chk_opt['hessian'].reshape(3*len(numbers),3*len(numbers))
energy_opt = chk_opt['ene']
rvecs_opt = chk_opt['rvecs']

mol_opt = Molecule(numbers, pos, masses, energy_opt, gradient, hessian_opt)
nma_opt = NMA(mol_opt)

freqs_opt = nma_opt.freqs[3:]
vol_opt = np.abs(np.dot(np.cross(rvecs_opt[0,:],rvecs_opt[1,:]),rvecs_opt[2,:]))

#load other volumes
structures = np.array([101,102,103,104,105,106])
freqs = np.zeros((len(structures),len(freqs_opt)))
energies = np.zeros(len(structures))
volumes = np.zeros(len(structures))
for i in range(len(structures)):
    chk = load_chk('{}/vasp.chk'.format(structures[i]))
    hessian = chk['hessian'].reshape(3*len(numbers),3*len(numbers))
    ene = chk['ene']
    rvecs = chk['rvecs']

    molec = Molecule(numbers, pos, masses, ene, gradient, hessian)
    nma = NMA(molec)

    freqs[i] = nma.freqs[3:]
    energies[i] = ene
    volumes[i] = np.abs(np.dot(np.cross(rvecs[0,:],rvecs[1,:]),rvecs[2,:]))

#plot properties

plot_free_energy(volumes,energies,freqs,degree=5,Temp=[1,100,200,300,400,500])
plot_properties(volumes,energies,freqs,masses,degree=5,Temp=np.arange(1,605,25))

