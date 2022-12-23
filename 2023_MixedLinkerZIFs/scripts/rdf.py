#!/usr/bin/env python

import numpy as np
import h5py, os
from yaff import *
from yaff_rdf_CN import *
from molmod.io.xyz import XYZFile
from molmod.periodic import periodic as pt
from molmod.units import *
from molmod.molecular_graphs import MolecularGraph
from molmod.molecular_graphs import HasAtomNumber, HasNumNeighbors, HasNeighborNumbers, HasNeighbors
from molmod.graphs import CritAnd, CritNot, CritOr
#import matplotlib.pyplot as pl

#log.set_level(log.silent)

C_Me1 = CritAnd(HasAtomNumber(6), HasNeighborNumbers(1,1,1,6))
H_Me = CritAnd(HasAtomNumber(1), HasNeighbors(C_Me1))
C_Me2 = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(7),HasAtomNumber(7),C_Me1))
N_Me = CritAnd(HasAtomNumber(7), HasNeighbors(HasAtomNumber(6),HasAtomNumber(30),C_Me2))
C_Me3 = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1),HasAtomNumber(6),N_Me))
H_Im_Me = CritAnd(HasAtomNumber(1), HasNeighbors(C_Me3))

C_HCO1 = CritAnd(HasAtomNumber(6), HasNeighborNumbers(1,6,8))
O_HCO = CritAnd(HasAtomNumber(8), HasNeighborNumbers(6))
H_HCO = CritAnd(HasAtomNumber(1), HasNeighbors(C_HCO1))
C_HCO2 = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(7),HasAtomNumber(7),C_HCO1))
N_HCO = CritAnd(HasAtomNumber(7), HasNeighbors(HasAtomNumber(6),HasAtomNumber(30),C_HCO2))
C_HCO3 = CritAnd(HasAtomNumber(6), HasNeighbors(HasAtomNumber(1),HasAtomNumber(6),N_HCO))
H_Im_HCO = CritAnd(HasAtomNumber(1), HasNeighbors(C_HCO3))

afilters = [('H_Me',H_Me),('H_Im_Me',H_Im_Me),('H_HCO',H_HCO),('H_Im_HCO',H_Im_HCO)]

class CP2K_xyz(object):

    def __init__(self, fn_prefix, directory='.',nvt_cell=np.eye(3)):

        from molmod.io.xyz import XYZFile

        xyz = XYZFile('{}/{}-pos-1.xyz'.format(directory, fn_prefix))
        self.geometries = xyz.geometries
        self.numbers = xyz.numbers
        self.symbols = xyz.symbols

        self.cell = np.tile(nvt_cell, (len(self.geometries),1,1))*angstrom
        print self.cell
        print np.shape(self.cell)

def get_hydrogen_types(num,pos,rvecs):

    sys = System(num,pos,rvecs=rvecs)
    bond_dist = {(1,6):1.5*angstrom,(1,8):1.5*angstrom,(6,6):1.8*angstrom,(6,7):2.0*angstrom,(6,8):1.8*angstrom,(7,30):2.5*angstrom}
    sys.detect_bonds(exceptions=bond_dist)

    graph = MolecularGraph(sys.bonds, sys.numbers)
    Htypes = [0]*len(sys.numbers)
    teller = 0
    for Htype, filter in afilters:
        print Htype
        for iatom, number in enumerate(sys.numbers):
            if filter(iatom, graph):
                Htypes[iatom] = Htype
                teller += 1
                print Htype, iatom
    print teller

    return Htypes

def make_rdf(fn_traj, rcut, rspacing, out_name, N_MOF=0):

    '''
        Make RDF for oxygen and four types of hydrogen atoms.

            rcut: in angstrom
            rspacing: in angstrom
            N_MOF: the first N_MOF atoms are not taken into account when constructing the RDF (convenient when studying for instance H2O adsorbed in a MOF)
    '''

    start_analysis = 0
    end_analysis = -1

    f = h5py.File(fn_traj, 'a')
    numbers = np.array(f['system/numbers'])
    positions = np.array(f['system/pos'])
    cell = np.array(f['system/cell'])
    Htypes = get_hydrogen_types(numbers,positions,cell)

    select_O = []
    select_H = []
    select_H_Me = []
    select_H_Im_Me = []
    select_H_HCO = []
    select_H_Im_HCO = []
    for i, num in enumerate(numbers):
        if num == 8 and i >= N_MOF:
            select_O.append(i)
        elif num == 1 and i >= N_MOF:
            select_H.append(i)
        if Htypes[i] == 'H_Me' and i >= N_MOF:
            select_H_Me.append(i)
        elif Htypes[i] == 'H_Im_Me' and i >= N_MOF:
            select_H_Im_Me.append(i)
        elif Htypes[i] == 'H_HCO' and i >= N_MOF:
            select_H_HCO.append(i)
        elif Htypes[i] == 'H_Im_HCO' and i >= N_MOF:
            select_H_Im_HCO.append(i)

    # Compute RDF
    rdf_H = RDF(rcut*angstrom, rspacing*angstrom, f, start=start_analysis, end=end_analysis, select0=select_O, select1=select_H, nimage=1, cellpath='trajectory/cell', outpath='rdf_H')
    rdf_H_HCO = RDF(rcut*angstrom, rspacing*angstrom, f, start=start_analysis, end=end_analysis, select0=select_O, select1=select_H_HCO, nimage=1, cellpath='trajectory/cell', outpath='rdf_H_HCO')
    rdf_H_Im_HCO = RDF(rcut*angstrom, rspacing*angstrom, f, start=start_analysis, end=end_analysis, select0=select_O, select1=select_H_Im_HCO, nimage=1, cellpath='trajectory/cell', outpath='rdf_H_Im_HCO')

    # Write out a data file containing the radial distance and the values of the RDF g(r)
    g = open('RDF_' + 'OH' + '.dat', 'w')
    g.write('# Distance (A) \tRDF \tH \t H_HCO \t H_Im_HCO \n')
    for i in range(len(rdf_H.d)):
        g.write(str(rdf_H.d[i]/angstrom) + "\t" + str(rdf_H.rdf[i]) + "\t" + str(rdf_H_HCO.rdf[i]) + "\t" + str(rdf_H_Im_HCO.rdf[i]) + "\t" + str(rdf_H.CN[i]) + "\n")
    g.close()

    #pl.plot(rdf.d/angstrom, rdf.rdf)
    #pl.show()

def calc_rdf_CP2K(fn_prefix, rcut=17, rspacing=0.01, N_MOF=0, fn_output='', directory='.', cell=np.eye(3),start=0):

    '''
        Make RDF for oxygen and four types of hydrogen atoms.

            rcut: RDF cutoff (in angstrom)
            rspacing: RDF spacing data points (in angstrom)
            N_MOF: the first N_MOF atoms are not taken into account when constructing the RDF (convenient when studying for instance H2O adsorbed in a MOF)
            directory: folder of the CP2K XYZ files
    '''

    # Calculate RDF
    xyz = CP2K_xyz(fn_prefix, directory=directory, nvt_cell=cell)

    with h5py.File('traj_rdf.h5', 'w') as h:
        h['system/numbers'] = xyz.numbers
        h['system/pos'] = xyz.geometries[start, :, :]
        h['system/cell'] = xyz.cell[start, :, :]
        h['trajectory/pos'] = xyz.geometries[start:,:,:]
        h['trajectory/cell'] = xyz.cell[start:,:,:]

    make_rdf('traj_rdf.h5', rcut, rspacing, fn_output, N_MOF=N_MOF)
    os.remove('traj_rdf.h5')


if __name__=='__main__':

    directory = 'NVT/'
    fn_prefix = 'ZIF90'
    cell = np.diag([17.2522719177,17.2522719177,17.2522719177])

    calc_rdf_CP2K(fn_prefix,directory=directory,cell=cell,start=5000)
