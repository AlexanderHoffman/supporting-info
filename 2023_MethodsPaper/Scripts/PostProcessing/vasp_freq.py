#!/usr/bin/env python

#Extract the Hessian from parallel selective dynamics simulations

from molmod.units import centimeter, electronvolt, angstrom, amu
from molmod.constants import lightspeed, planck
from molmod.unit_cells import UnitCell
from molmod import angstrom, lightspeed, centimeter
from molmod.io import XYZWriter as MMXYZWriter
from molmod.periodic import periodic
import numpy as np
import os
from copy import deepcopy
from yaff import *
log.set_level(log.silent)
from myxml import *
from molmod.io import dump_chk, load_chk

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
            'hessian'   : (".//dynmat/varray[@name='hessian']", -electronvolt/angstrom**2/amu),
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
        for iatype in range(masses.shape[0]):
            self.fields['masses'][atomtypes==iatype] = masses[iatype]
        # Read SCF energies
        self.fields['energies'] = np.array([float(step.find('energy/i[@name="e_fr_energy"]').text)*electronvolt\
                 for step in self.root.findall('.//calculation')])
        # Read all requested arrays
        for label in field_labels:
            if not label in tag_dictionary.keys():
                raise NotImplementedError("Failed to read %s from xml file" % label)
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
                   
def read_abinitio(fn, workdir, natoms, nfree=2, potim=0.015, fn_chk='vasp.chk'):
    ''' TAKEN FROM QUICKFF
    
        Wrapper to read all information from an ab initio calculation that
        QuickFF needs. Currently Gaussian .fchk and VASP .xml files are
        supported.
    '''
    extension = fn.split('.')[-1]
    if extension=='xml':
        vasprun = VASPRun(fn,field_labels=['gradient'])
        numbers = vasprun.fields['numbers']
        coords = vasprun.fields['pos_init']
        energy = vasprun.fields['energies'][0]
        grad = -vasprun.fields['gradient'][0]
        hess = combine_hessians(workdir, natoms, nfree=nfree, potim=potim)
        #hess = vasprun.fields['hessian'].reshape((len(numbers),3,len(numbers),3 ))
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
    
def read_atominfo(xml_fn):
    fxml = open( xml_fn, 'r')
    #Get the ionic optimization steps
    atominfo = getBlocs("atominfo", fxml)[0]
    atomtypes = []
    atomtype_ids = []
    for line in atominfo:
        data = getNodeData(line)
        if len(data)==2:
            try: atomtype_ids.append( int(data[1]) )
            except: pass
        if len(data)==6:
            try: atomtypes.append( [float(data[1])*amu,data[4]] )
            except: pass
    atomnumbers = np.zeros( (len(atomtype_ids),  ) , dtype='i4')
    atommasses = np.zeros( (len(atomtype_ids),  ) )
    for i,atomtype in enumerate(atomtype_ids):
        print(atomtypes[atomtype-1][1][:-3])
        if atomtypes[atomtype-1][1][:-3]=='V_sv':
            atomnumbers[i] = periodic['V'].number
        elif atomtypes[atomtype-1][1][:-3]=='Co_sv':
            atomnumbers[i] = periodic['Co'].number  
        elif atomtypes[atomtype-1][1][:-3]=='Zn_sv':
            atomnumbers[i] = periodic['Zn'].number         
        else:
            atomnumbers[i] = periodic[atomtypes[atomtype-1][1][:-3]].number
        atommasses[i] = atomtypes[atomtype-1][0]
    return atomnumbers, atommasses

def read_free_energies(xml_fn):
    '''
    Return the free energy for all ionic optimization steps from a vasprun
    xml_fn: the filename of the xml from the vasprun
    '''
    fxml = open( xml_fn, 'r')
    #Get the ionic optimization steps
    steps = getBlocs("calculation", fxml)
    energies = np.zeros( (len(steps), ) )
    arrays = {"basis":[],"forces":[],"positions":[]}
    units = {"basis":angstrom,"forces":electronvolt/angstrom,"positions":1.0}
    for i,step in enumerate(steps):
        current_array = None
        #Find the final energy for this ionic opt step
        for line in step:
            name = getClefs(line,["name"])["name"]
            if name == "e_fr_energy": 
                energy = float( getNodeData(line)[0])*electronvolt
                continue
            if name in arrays.keys(): 
                current_array = name
                arrays[name].append([])
                continue
            #Check if we finished reading an array
            if line=='</varray>': current_array = None
            #Read array elements if appropiate
            if current_array is None: continue
            #Append array elements
            for el in line.split(' '):
                try: arrays[current_array][-1].append(float(el))
                except: pass
        energies[i] = energy
        for array in arrays.keys():          
            arrays[array][-1] = np.reshape( np.asarray( arrays[array][-1] ) , (-1,3) ) * units[array]
        arrays["basis"][-1] = np.transpose( arrays["basis"][-1] )
        #print arrays["basis"][-1]
        #Convert fractional to cartesian coordinates    
        arrays["positions"][-1] = np.dot( arrays["positions"][-1] , arrays["basis"][-1] )
        arrays["forces"][-1] = -arrays["forces"][-1]
    return energies,arrays["basis"],arrays["forces"],arrays["positions"]

def combine_hessians(workdir, natoms, nfree=2, potim=0.015):
    '''
    Combine displacements of vaspruns per atom to obtain full hessian
    
    natoms = number of atoms in the unit cell (see POSCAR)
    
    nfree = number of atomic displacements (see INCAR)
    
    potim = displacement in angstrom (see INCAR)
    
    '''
    ndof = 3*natoms
    stepsize = potim*angstrom
    #Setup force matrix, a [nfree x ndof x ndof] matrix
    #Entry l,i,j gives the force on ndof j when for the l'th displacement of ndof i
    F = np.zeros( (nfree,ndof,ndof) ) 
    
    for a in range(natoms):
    
        print(a)
        #Read info from atomic displacement
        energies, rvecs, gradients, positions = read_free_energies(os.path.join(workdir,'%s'%a,'vasprun.xml'))
        #assert len(gradients)== (3*nfree*natoms + 1)
        for b in range(3*nfree):
            i = int(3*a + np.floor( b/nfree )) #Which dof is displaced
            l = b%nfree #How many displacements
            F[l,i,:] = np.reshape(gradients[b+1],(-1,))

    #Construct the Hessian if you only take 2 steps
    H = F[0,:,:] - F[1,:,:]
    #This is the mass weighted hessian, vasprun.xml contains the mass-unweighted hessian
    hessian = (H + np.transpose(H))/(4.0*stepsize) 

    return hessian
    
if __name__=='__main__':
    read_abinitio('0/vasprun.xml', '.', natoms=456)
