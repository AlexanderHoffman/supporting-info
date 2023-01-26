#! /usr/bin/env python
# This script contains some methods to extract raman spectra from
# ab initio molecular dynamics runs (with CP2K). Code for the Fourier-transform
# of autocorrelations comes from MD-tracks.
# Atomic units everywhere, except for providing output (where units are always mentioned)

import os, sys
import numpy as np
import matplotlib.pyplot as pt
import cPickle as pickle

from molmod.io import XYZReader, XYZWriter
from molmod.units import *
from molmod.constants import *

from yaff.pes.ext import Cell

class SpectrumProcessor(object):
    """Implements a stepwise implementation for the computation of spectra.

       Each input from a multidimensional trajectory is processed separately.
       One does not have to load the entire trajectory in memory at the same
       time.
    """

    def __init__(self, time_step, num_blocks=10):
        """Initialize a spectrum SpectrumProcessor instance.

           Arguments:
             time_step  --  The time step [au] of the inputs that will be given
                            to the process method.
             num_blocks  --  The number of blocks in which the inputs will be
                             divided. A Fourier transform of each block is computed
                             and the final spectrum is the average over all blocks.
                             [default=10] A larger number of blocks implies a lower
                             resolution on the frequency scale, but yields an
                             improved statistical accuracy of the amplitudes.
        """
        self._time_step = time_step
        self._num_blocks = num_blocks
        self._sum = 0
        self._sum_sq = 0
        self._count = 0

        self._input_length = None
        self._block_size = None

    block_size = property(lambda self: self._block_size)

    def process(self, fn):
        """Add an input to the spectrum.

           Arguments:
             fn  --  A one-dimensional array with function values taken at equi-
                     distant time steps. (See the time_step argument of the
                     __init__ method.)
        """
        if len(fn) < 2*self._num_blocks:
            raise ValueError("The length of the input for the spectrum must at least be two times the block size.")
        if self._input_length is None:
            self._input_length = len(fn)
            self._block_size = len(fn)/self._num_blocks
        elif self._input_length != len(fn):
            raise ValueError("All inputs must have the same length.")

        for index in xrange(self._num_blocks):
            fn_block = fn[index*self._block_size:(index+1)*self._block_size]
            amp_block = abs(np.fft.rfft(fn_block))**2
            self._sum += amp_block
            self._sum_sq += amp_block**2
            self._count += 1

    def get_results(self):
        """Compute the resulting spectrum.

           Returns a tuple with four elements:
             frequency_res -- The resolution of the spectrum on the frequency
                              scale.
             wavenumber_res  --  The resulution of the spectrum on the
                                 wavenumber scale.
             amplitudes  --  An array of amplitudes representing the spectrum.
             amplitudes_err  --  The statistical error on the amplitudes.
        """
        if self._count == 0:
            raise RuntimeError("There are no results yet. Call the process method first.")

        amplitudes = self._sum/(self._count*self._block_size)
        amplitudes_sq = self._sum_sq/(self._count*self._block_size)
        amplitudes_err = np.sqrt((amplitudes_sq - amplitudes**2))/self._count # error on the mean

        duration = self._time_step*self._block_size
        frequency_res = 1.0/duration
        wavenumber_res = 1.0/duration/lightspeed

        return frequency_res, wavenumber_res, amplitudes, amplitudes_err

def polarizability(E_0,E_x,E_y,E_z,cell=None):
    dip_E = []
    dip_E.append(E_x)
    dip_E.append(E_y)
    dip_E.append(E_z)
    dip_E = np.asarray(dip_E)
    print cell
    for i in xrange(len(dip_E)):
        print i
        for j in xrange(len(dip_E[i])):
            value = dip_E[i,j,:] - E_0[j,:] 
            ref = value.copy()
            rvecs = Cell(np.array([cell[:,0],cell[:,1],cell[:,2]]))
            rvecs.mic(value)
            disp = value-ref
            dip_E[i,j,:] += disp

    polar = []                        
    for direction in dip_E:
        for i in xrange(3):
            polar.append(direction[:,i] - E_0[:,i])
    step = np.arange(len(polar[0]))

    pt.plot(step,polar[0],label='xx')
    pt.plot(step,polar[1],label='xy')
    pt.plot(step,polar[2],label='xz')
    pt.plot(step,polar[3],label='yx')
    pt.plot(step,polar[4],label='yy')
    pt.plot(step,polar[5],label='yz')
    pt.plot(step,polar[6],label='zx')
    pt.plot(step,polar[7],label='zy')
    pt.plot(step,polar[8],label='zz')
    pt.legend()
    pt.show()

    alpha = []
    for i in xrange(len(polar[0])):
        alpha.append([polar[0][i],polar[4][i],polar[8][i],polar[1][i],polar[2][i],polar[5][i],polar[3][i],polar[6][i],polar[7][i]]) 
        if np.any(np.array(alpha[i])-np.array(alpha[0])>500):
            print i
            print np.array(alpha[i])
            print E_0[i],E_x[i],E_y[i],E_z[i]
    return alpha

def deriv_polar(alpha,timestep=2.0*femtosecond):
    deriv_alpha = np.zeros(np.shape(alpha))
    for i in xrange(len(alpha)):
        if i>0:
            deriv_alpha[i,:] = (alpha[i-1,:] - alpha[i,:])/timestep
    return deriv_alpha

def raman_spectrum(alpha, timestep=2.0*femtosecond, num_blocks = 8, k_in=400.0*lightspeed/centimeter, T=300.0*kelvin):
    
    #alpha with terms [xx,yy,zz,xy,xz,yz,yx,zx,zy]
    
    alpha = np.asarray(alpha)
    #alpha = deriv_polar(alpha)

    #isotropic part
    alpha_iso = []
    for i in xrange(len(alpha)):
        alpha_iso.append((alpha[i][0] + alpha[i][1] + alpha[i][2])/3.0)
    sp = SpectrumProcessor(timestep, num_blocks=num_blocks)
    sp.process(alpha_iso)   
    freq_res, wave_res, amps_iso, amp_err_iso = sp.get_results()
    
    #anisotropic part
    sp = SpectrumProcessor(timestep, num_blocks=num_blocks)
    
    aniso_parts=[3,4,5]    
    for aniso_part in aniso_parts:
        alpha_aniso = []
        for i in xrange(len(alpha)):
            alpha_aniso.append(np.sqrt(3.0)*(alpha[i][aniso_part]))
        sp.process(alpha_aniso)    
    
    aniso_parts=[[0,1],[1,2],[2,0]]
    for aniso_part in aniso_parts:
        alpha_aniso = []
        for i in xrange(len(alpha)):
            alpha_aniso.append(np.sqrt(0.5)*( alpha[i][aniso_part[0]] - alpha[i][aniso_part[1]]))
        sp.process(alpha_aniso)
    
    freq_res, wave_res, amps_aniso, amp_err_aniso = sp.get_results()
    
    # Get array with frequencies
    freqs = freq_res*np.arange(amps_iso.shape[0])
    # Remove frequency zero to avoid problems with prefactor
    amps_iso = amps_iso[np.where(freqs/lightspeed*centimeter>1)]
    amps_aniso = amps_aniso[np.where(freqs/lightspeed*centimeter>1)]
    freqs = freqs[np.where(freqs/lightspeed*centimeter>1)]

    # Multiply with prefactor
    k_in = lightspeed/(676.4*nanometer)
    prefactor = (k_in-freqs)**4*freqs/(1.0-np.exp(-freqs/boltzmann/T))
    amps_iso *= prefactor
    amps_aniso *= prefactor
    # Compute intensity of para and ortho polarized light
    amps_para = (45.0*amps_iso + 4.0*amps_aniso)/45.0 
    amps_ortho = amps_aniso / 15.0
    return freqs, amps_para, amps_ortho

dip_E0 = np.genfromtxt('../dipole/E0/dipole.txt')[:8000,:]
dip_Ex = np.genfromtxt('../dipole/Ex/dipole.txt')[:8000,:]
dip_Ey = np.genfromtxt('../dipole/Ey/dipole.txt')[:8000,:]
dip_Ez = np.genfromtxt('../dipole/Ez/dipole.txt')[:8000,:]
print len(dip_Ez)

cell_vec = np.genfromtxt('../dipole/cell.txt')

alpha = polarizability(dip_E0,dip_Ex,dip_Ey,dip_Ez,cell=cell_vec)

freq, amps_para, amps_ortho = raman_spectrum(alpha,timestep=10.0*femtosecond,num_blocks=1,T=100.0*kelvin)
amps = amps_para + amps_ortho

freq = freq/lightspeed*centimeter

mask = freq>=0
amps *= mask

np.savetxt('raman_intensities', np.c_[freq, amps])


        
