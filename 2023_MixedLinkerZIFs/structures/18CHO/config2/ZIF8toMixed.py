#! /usr/bin/env python

import numpy as np
from yaff import *
from tamkin import *
from molmod import *

def make_graph(edges, crit, bonds, numbers):

    pattern_graph = Graph(edges)
    criteria = CriteriaSet(vertex_criteria=crit)
    pattern = CustomPattern(pattern_graph, criteria_sets=[criteria])
    
    graph = MolecularGraph(bonds, numbers)
    gs = GraphSearch(pattern)    
    atom_list = []
    for match in gs(graph):
        pattern_atoms = match.forward
        atom_list.append([pattern_atoms[key] for key in pattern_atoms])

    return atom_list

def edges_linker(atom,count):
    return [(atom+count+0,atom+count+1),(atom+count+0,atom+count+2),(atom+count+0,atom+count+3),(atom+count+0,atom+count+4),(atom+count+4,atom+count+5),(atom+count+4,atom+count+6)]

def ZIF8_linker(atom):
    
    graph_edges = edges_linker(atom,0)

    criteria = {atom+0 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 6)), atom+1 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+2 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+3 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+4 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7)), atom+5 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+6 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30))}
    
    return graph_edges, criteria

def ZIF8_4ring(atom):
    
    graph_edges = edges_linker(atom,0)+edges_linker(atom,8)+edges_linker(atom,16)+edges_linker(atom,24)+[(atom+7,atom+6),(atom+7,atom+13),(atom+15,atom+14),(atom+15,atom+21),(atom+23,atom+22),(atom+23,atom+29),(atom+31,atom+30),(atom+31,atom+5)]

    criteria = {atom+0 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 6)), atom+1 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+2 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+3 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+4 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7)), atom+5 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+6 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+7 : CritAnd(HasAtomNumber(30), HasNeighborNumbers(7, 7, 7, 7)),atom+8 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 6)), atom+9 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+10 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+11 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+12 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7)), atom+13 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+14 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+15 : CritAnd(HasAtomNumber(30), HasNeighborNumbers(7, 7, 7, 7)), atom+16 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 6)), atom+17 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+18 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+19 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+20 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7)), atom+21 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+22 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+23 : CritAnd(HasAtomNumber(30), HasNeighborNumbers(7, 7, 7, 7)), atom+24 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 6)), atom+25 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+26 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+27 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+28 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7)), atom+29 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+30 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+31 : CritAnd(HasAtomNumber(30), HasNeighborNumbers(7, 7, 7, 7))}
    
    return graph_edges, criteria

#Read chk
sys = System.from_file('ZIF8.chk')
bond_dist = {(1,6):1.5*angstrom,(6,6):1.5*angstrom,(6,8):1.5*angstrom,(6,7):1.5*angstrom,(7,30):2.0*angstrom}
sys.detect_bonds(exceptions=bond_dist)
bonds = sys.bonds
num = sys.numbers
pos = sys.pos
rvecs = sys.cell.rvecs

#Find linkers
linker_edges = []
linker_criteria = {}    

data = ZIF8_linker(0)
linker_edges += data[0]
linker_criteria.update(data[1])
data[1].clear()

linker_atoms = np.array(make_graph(linker_edges, linker_criteria, bonds, num))
print linker_atoms

#Find 4-rings
ring_edges = []
ring_criteria = {}    

data2 = ZIF8_4ring(0)
ring_edges += data2[0]
ring_criteria.update(data2[1])
data2[1].clear()

ring_atoms_tmp = np.array(make_graph(ring_edges, ring_criteria, bonds, num))
ring_atoms = np.zeros((6,32))
counter = 0
for i in range(len(ring_atoms_tmp)):
    vec = pos[ring_atoms_tmp[i,0],:]-pos[ring_atoms_tmp[i,16],:]
    sys.cell.mic(vec)
    if np.linalg.norm(vec)<5*angstrom:
        ring_atoms[counter,:] = ring_atoms_tmp[i,:]
        counter += 1
print ring_atoms

Zn_atoms = ring_atoms[:,[7,15,23,31]]
group1 = []
group2 = []
for i in range(len(Zn_atoms)):
    add = True
    for j in range(len(Zn_atoms[i,:])):
        if (i > 0 and (Zn_atoms[i,j] in Zn_atoms[:i,:])):
            add = False
    if add:
        group1.append(i)
    else:
        group2.append(i)

group_rings = np.asarray([group1,group2])
print group_rings

#Define new structure     
num2 = []
pos2 = []
masses2 = []
for i in range(len(num)):
    added = False
    for j in range(len(linker_atoms)):
        adapt = True
        print ring_atoms[group_rings[1,:],0:8]
        print linker_atoms[j]
        if (all(elem in ring_atoms[group_rings[1,:],0:8] for elem in linker_atoms[j]) or all(elem in ring_atoms[group_rings[1,:],16:24] for elem in linker_atoms[j])):
            adapt = False
        print adapt
        if (i==linker_atoms[j][1] and adapt):
            vec = pos[linker_atoms[j][4]]-pos[linker_atoms[j][5]]
            sys.cell.mic(vec)    
            new_pos = (pos[linker_atoms[j][0]]+vec*1.21*angstrom/np.linalg.norm(vec))
            num2.append(8)
            pos2.append(new_pos)   
            masses2.append(periodic.periodic[8].mass) 
            added = True             
        elif (i==linker_atoms[j][2] and adapt):
            vec = pos[linker_atoms[j][4]]-pos[linker_atoms[j][6]]
            sys.cell.mic(vec) 
            new_pos = (pos[linker_atoms[j][0]]+vec*1.09*angstrom/np.linalg.norm(vec))
            num2.append(1)
            pos2.append(new_pos)
            masses2.append(periodic.periodic[1].mass)  
            added = True
        elif (i==linker_atoms[j][3] and adapt):
            added = True              
    if not added:
        num2.append(num[i])
        pos2.append(pos[i])
        masses2.append(sys.masses[i])  

num2 = np.asarray(num2)
pos2 = np.asarray(pos2)
masses2 = np.asarray(masses2)
            
#Write chk
sys2 = System(num2, pos2, rvecs=rvecs, masses=masses2)
sys2.to_file('ZIF8_18CHO.chk')
