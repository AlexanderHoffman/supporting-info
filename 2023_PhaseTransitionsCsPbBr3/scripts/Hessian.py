#!/usr/bin/env python

from tamkin import NMA, Molecule,ConstrainExt
from molmod.units import centimeter, electronvolt, angstrom, amu
from molmod.constants import lightspeed, planck
from molmod.unit_cells import UnitCell
from molmod import angstrom, lightspeed, centimeter
from molmod.io import XYZWriter as MMXYZWriter
from molmod.periodic import periodic
import numpy as np
#from vasp_tamkin import load_molecule_vasp
import os
from copy import deepcopy
from yaff import *
log.set_level(log.silent)
import cPickle as pickle
from molmod.io import dump_chk, load_chk

from tamkin.io import load_molecule_vasp, dump_modes_xyz
from tamkin import ConstrainExt, NMA
import argparse
import os
import os.path

import xml.etree.ElementTree as ET

class VASPRun(object):

    ''' TAKEN FROM QUICKFF
    
        Load information from a vasprun.xml file
    '''
    def __init__(self, filename, field_labels=[]):
        '''
            **Arguments**

            filename
                Filename of vasprun.xml

            **Optional Arguments**

            field_labels
                List of things we want to read. If an empty list is provided,
                only numbers, masses, initial positions and initial cell
                vectors are read.
        '''
        # Link between field labels and tags in the xml file
        tag_dictionary = {
            'rvecs_init': (".//structure[@name='initialpos']/crystal/varray[@name='basis']", angstrom),
            'pos_init'  : (".//structure[@name='initialpos']/varray[@name='positions']", 1.0),
            'gradient'  : (".//varray[@name='forces']", -electronvolt/angstrom),
            'hessian'   : (".//dynmat/varray[@name='hessian']", -(hertz*10**12*planck)/angstrom**2/amu),
                }
        self.fields = {}
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        assert self.root.tag=='modeling', "Root tag is not modeling, this is not a standard vasprun.xml file"
        if not 'rvecs_init' in field_labels: field_labels.append('rvecs_init')
        if not 'pos_init' in field_labels: field_labels.append('pos_init')
        # Read atomic numbers
        self.fields['numbers'] = np.asarray([periodic[atom.find('c').text.strip()].number for atom in self.root.findall(".//*[@name='atoms']/set")[0]])
        # Read atomtypes
        atomtypes = np.asarray([int(atom.findall('c')[1].text) for atom in self.root.findall(".//*[@name='atoms']/set")[0]]) - 1
        # Read atomic masses for atomtype
        masses = np.asarray([float(atom.findall('c')[2].text) for atom in self.root.findall(".//*[@name='atomtypes']/set")[0]])*amu
        self.fields['masses'] = np.zeros(self.fields['numbers'].shape)
        for iatype in xrange(masses.shape[0]):
            self.fields['masses'][atomtypes==iatype] = masses[iatype]
        # Read SCF energies
        self.fields['energies'] = np.array([float(step.find('energy/i[@name="e_fr_energy"]').text)*electronvolt\
                 for step in self.root.findall('.//calculation')])
        # Read all requested arrays
        for label in field_labels:
            if not label in tag_dictionary.keys():
                raise NotImplementedError, "Failed to read %s from xml file" % label
            self.fields[label] = self.read_array(tag_dictionary[label][0], unit=tag_dictionary[label][1])
        # Convert fractional to Cartesian coordinates
        self.fields['pos_init'] = np.dot(self.fields['pos_init'], self.fields['rvecs_init'])
        # Hessian is mass-weighted, we want the pure second derivatives
        if 'hessian' in self.fields.keys():
            m3 = np.sqrt(np.array(sum([[m,m,m] for m in self.fields['masses']],[])))
            self.fields['hessian'] = m3.reshape((-1,1))*self.fields['hessian']*m3
            
    def read_array(self, tag, unit=1.0):
        result = []
        for match in self.root.findall(tag):
            result.append([])
            for line in match.findall('v'):
                result[-1].append([float(w) for w in line.text.split()])
        if len(result)==1: result = result[0]
        return np.asarray(result)*unit
                   
def read_abinitio(fn, workdir, nfree=2, potim=0.015, fn_chk='vasp.chk'):
    ''' TAKEN FROM QUICKFF
    
        Wrapper to read all information from an ab initio calculation that
        QuickFF needs. Currently Gaussian .fchk and VASP .xml files are
        supported.
    '''
    extension = fn.split('.')[-1]
    if extension=='xml':
        vasprun = VASPRun(fn,field_labels=['gradient','hessian'])
        numbers = vasprun.fields['numbers']
        coords = vasprun.fields['pos_init']
        energy = vasprun.fields['energies'][0]
        grad = -vasprun.fields['gradient'][0]
        hess = vasprun.fields['hessian'].reshape((len(numbers),3,len(numbers),3 ))
        masses = vasprun.fields['masses']
        rvecs = vasprun.fields['rvecs_init']
        pbc = [1,1,1]
    else: raise NotImplementedError

    dump_chk(fn_chk, {
    'numbers': numbers,
    'pos': coords,
    'ene': energy,
    'rvecs': rvecs,
    'masses': masses,
    'hessian': hess.reshape([len(numbers), 3, len(numbers), 3]),
    'gradient': grad.reshape([len(numbers), 3])
    })
    
    return
    
if __name__=='__main__':
    read_abinitio('vasprun.xml', '.',fn_chk='vasp.chk')
