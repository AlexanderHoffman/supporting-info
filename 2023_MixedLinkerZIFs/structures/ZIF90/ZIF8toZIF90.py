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

def ZIF8_linker(atom):
    
    graph_edges = [(atom+0,atom+1),(atom+0,atom+2),(atom+0,atom+3),(atom+0,atom+4),(atom+4,atom+5),(atom+4,atom+6)]

    criteria = {atom+0 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(1, 1, 1, 6)), atom+1 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+2 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+3 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), atom+4 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 7, 7)), atom+5 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30)), atom+6 : CritAnd(HasAtomNumber(7), HasNeighborNumbers(6, 6, 30))}
    
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
print np.shape(num), np.shape(pos)

#Define new linkers        
num2 = []
pos2 = []
masses2 = []
for i in range(len(num)):
    added = False
    for j in range(len(linker_atoms)):
        if i==linker_atoms[j][1]:
            vec = pos[linker_atoms[j][4]]-pos[linker_atoms[j][5]]
            sys.cell.mic(vec)    
            print pos[linker_atoms[j][0]]/angstrom/16.95398
            print vec/angstrom/16.95398 
            new_pos = (pos[linker_atoms[j][0]]+vec*1.21*angstrom/np.linalg.norm(vec))
            print 1.21*angstrom/np.linalg.norm(vec)
            print new_pos/angstrom/16.95398
            num2.append(8)
            pos2.append(new_pos)   
            masses2.append(periodic.periodic[8].mass) 
            added = True             
        elif i==linker_atoms[j][2]:
            vec = pos[linker_atoms[j][4]]-pos[linker_atoms[j][6]]
            sys.cell.mic(vec) 
            new_pos = (pos[linker_atoms[j][0]]+vec*1.09*angstrom/np.linalg.norm(vec))
            num2.append(1)
            pos2.append(new_pos)
            masses2.append(periodic.periodic[1].mass)  
            added = True
        elif i==linker_atoms[j][3]:
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
sys2.to_file('ZIF90.chk')
