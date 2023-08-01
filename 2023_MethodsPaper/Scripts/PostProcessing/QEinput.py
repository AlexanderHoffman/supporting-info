#Auxiliary file for make_files_QE.py

import os
import shutil
import numpy as np
from molmod.units import *
from molmod.periodic import periodic
import xml.etree.ElementTree as ET
from string import Template

def parse_species(numbers):
    elements = np.unique(numbers)
    string = ''
    for i in range(len(elements)):
        string += '%s\t%s\t%s.pbe-paw.UPF\n' % (periodic[elements[i]].symbol, str(periodic[elements[i]].mass/amu), str(periodic[elements[i]].symbol))
    return string[:-1]
    
def parse_positions(numbers, positions):
    string = ''
    for i in range(positions.shape[0]):
        string += '%s\t%s\t%s\t%s\n' % (periodic[numbers[i]].symbol, str(positions[i, 0]), str(positions[i, 1]), str(positions[i, 2]))
    return string[:-1]
    
def parse_rvec(rvec):
    string = ''
    for i in range(3):
        string += '%s\t%s\t%s\n' % (str(rvec[i, 0]), str(rvec[i, 1]), str(rvec[i, 2]))
    return string[:-1]

class QEModel(object):

    def __init__(self, fn_in = 'input.xml', prefix = 'input', cutoff = 40, kpoints = (1,1,1)):
        self.input = fn_in
        self.prefix = prefix   
        self.cutoff = cutoff
        self.kpoints = kpoints
        if self.input.find('.xml') != -1:
            xml_data = QE_xml(self.input)
            self.num = xml_data.fields['num']
            self.pos = xml_data.fields['pos']/angstrom
            self.rvec = xml_data.fields['rvec']/angstrom
        elif self.input.find('POSCAR') != -1:
            POSCAR_data = QE_POSCAR(self.input)
            self.num = POSCAR_data.numbers
            self.pos = POSCAR_data.coords/angstrom
            self.rvec = POSCAR_data.rvecs/angstrom
        else:
            raise ValueError('Input file not found')
            
    def write_pw_input(self, fn='input.in', calculation = 'scf', conv_thr = 1e-09, efield = None):
        
        # Write the input file here       
        if efield is None:
            efield_control = ''
            efield_cart = ''
        else:
            efield_control = '''lelfield = .TRUE.
   nberrycyc = 1'''
            efield_cart = '''efield_cart(1) = %f
   efield_cart(2) = %f
   efield_cart(3) = %f''' % (efield[0], efield[1], efield[2])

        if calculation.find('relax') != -1:
            geo_opt_control = '''nstep         = 100
   etot_conv_thr = 1.0d-6'''
        else:
            geo_opt_control = ''
            
        exponent = int(np.floor(np.log10(conv_thr)))
        base = conv_thr / 10**exponent
        conv_thr_string = '%fD%d' % (base, exponent)
        
        species_string = parse_species(self.num)
        positions_string = parse_positions(self.num,self.pos)
        rvec_string = parse_rvec(self.rvec)       
        nat = self.pos.shape[0]
        ntyp = len(np.unique(self.num))
        
        kpoints_string = '''K_POINTS automatic
   %d %d %d   1 1 1''' % (self.kpoints[0], self.kpoints[1], self.kpoints[2])
        
        replacements = {'prefix' : self.prefix, 'calculation': calculation, 'geo_opt_control': geo_opt_control, 'efield_control' : efield_control, 'nat' : nat, 'ntyp' : ntyp, 'ecutwfc' : self.cutoff, 'conv_thr' : conv_thr_string, 'efield_cart' : efield_cart, 'species' : species_string, 'positions' : positions_string, 'kpoints' : kpoints_string, 'rvec' : rvec_string}
        
        with open('pw_template.in') as f:
            src = Template(f.read())
        result = src.substitute(replacements)
        
        with open('{}.in'.format(fn), 'w') as f:
            f.write(result) 

    def write_eos_input(self, fn, calculation = 'relax', conv_thr = 1e-09, volumes = [0.94,0.96,0.98,1.02,1.04,1.06]):

        efield_control = ''
        efield_cart = ''

        exponent = int(np.floor(np.log10(conv_thr)))
        base = conv_thr / 10**exponent
        conv_thr_string = '%fD%d' % (base, exponent)
        
        species_string = parse_species(self.num)     
        nat = self.pos.shape[0]
        ntyp = len(np.unique(self.num))
        
        kpoints_string = '''K_POINTS automatic
   %d %d %d   1 1 1''' % (self.kpoints[0], self.kpoints[1], self.kpoints[2])

        for i in range(len(volumes)):
            positions_string = parse_positions(self.num,volumes[i]**(1./3.)*self.pos)
            rvec_string = parse_rvec(volumes[i]**(1./3.)*self.rvec)  

            replacements = {'prefix' : self.prefix, 'calculation': calculation, 'efield_control' : efield_control, 'nat' : nat, 'ntyp' : ntyp, 'ecutwfc' : self.cutoff, 'conv_thr' : conv_thr_string, 'efield_cart' : efield_cart, 'species' : species_string, 'positions' : positions_string, 'kpoints' : kpoints_string, 'rvec' : rvec_string}
            
            with open('pw_template.in') as f:
                src = Template(f.read())
            result = src.substitute(replacements)
            
            with open('{}_{}.in'.format(fn,int(volumes[i]*100)), 'w') as f:
                f.write(result)     
            
    def read_output(self, fn):
        read_pos = False
        read_rvec = False
        read_forces = False
        read_stress = False
        positions = []
        forces = []
        stress = []
        
        with open(fn) as f:
            while True:
                line = f.readline()
                if '!    total energy' in line:
                    energy = float(line.split()[-2]) * rydberg
                    continue
                elif 'Begin final coordinates' in line:
                    read_pos = True
                    continue
                elif 'End final coordinates' in line:
                    read_pos = False
                    continue
                elif 'Forces acting on atoms' in line:
                    read_forces = True
                    continue
                elif 'Total force' in line:
                    read_forces = False
                    continue
                elif 'total   stress' in line:
                    read_stress = True
                    continue
                    
                if read_forces and 'atom' in line:
                    split = line.split()
                    forces.append([split[-3], split[-2], split[-1]])
                
                if read_stress:
                    if line == '\n':
                        read_stress = False
                    else:
                        split = line.split()
                        stress.append([split[0], split[1], split[2]])
                        
                if not line:
                    break
        
        positions = np.array(positions, dtype=np.float) * angstrom
        forces = np.array(forces, dtype=np.float) * rydberg
        stress = np.array(stress, dtype=np.float) * rydberg
        
        return energy, positions, forces, stress #in atomic units

class QE_xml(object):

    def __init__(self, fn, field_labels=[]):
        self.fields = {}
        self.root = ET.parse(fn).getroot()
        tag_dictionary = {'pos':'.//output/atomic_structure/atomic_positions','rvec':'.//output/atomic_structure/cell','forces':'.//output/forces'}

        #read rvecs
        rvecs = []
        for line in self.root.findall(tag_dictionary['rvec'])[-1]:
            rvecs.append([float(w) for w in line.text.split()])
        self.fields['rvec'] = np.asarray(rvecs)

        #read pos
        pos = []
        num = []
        for line in self.root.findall(tag_dictionary['pos'])[-1]:
            pos.append([float(w) for w in line.text.split()])
            num.append(line.attrib['name'])
        self.fields['pos'] = np.asarray(pos)
        self.fields['num'] = np.asarray(num)

        #read forces
        self.fields['forces'] = np.reshape(self.root.findall(tag_dictionary['forces'])[-1].text.split(),(-1,3)).astype(np.float)

class QE_POSCAR(object):

    def __init__(self,fn):
        f = open(fn, 'r')
        title = f.readline()
        scale = float(f.readline())
        self.rvecs = np.zeros([3,3], float)
        for i in range(3):
            line = f.readline()
            self.rvecs[i,:] = np.array([float(word)*scale*angstrom for word in line.split()])
        symbols = f.readline().split()
        ndups = [int(num) for num in f.readline().split()]
        natoms = sum(ndups)
        num = []
        for symbol, ndup in zip(symbols, ndups):
            for i in range(ndup):
                num.append(symbol)#periodic[symbol].number)
        self.numbers = np.array(num)
        title2 = f.readline().rstrip().lstrip()
        if title2.lower() in ['direct', 'cartesian']:
            mode = title2
        else:
            mode = f.readline().rstrip().lstrip()
        assert mode.lower() in ['direct', 'cartesian'], 'Error reading POSCAR, no Coordinate mode (direct or cartesian) found.'
        self.coords = np.zeros([natoms, 3], float)
        for i in range(natoms):
            line = f.readline()
            if mode.lower() == 'direct':
                frac = np.array([float(s) for s in line.split()[:3]])
                coord = np.dot(self.rvecs.T, frac)
            elif mode.lower() == 'cartesian':
                coord = np.array([float(s)*scale*angstrom for s in line.split()[:3]])
            else:
                raise ValueError('Invalid Coordinate mode specifier, recieved %s' %mode)
            self.coords[i,:] = coord.copy()

