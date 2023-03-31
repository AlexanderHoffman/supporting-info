#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as pt
import matplotlib.gridspec as gridspec
from yaff import *
from tamkin import *
from tamkin.io import *
from molmod.units import *
from molmod.constants import *
from molmod.periodic import periodic
from molmod.graphs import Graph, GraphSearch, CustomPattern, CritAnd, CriteriaSet, CritOr
from molmod.molecular_graphs import *
from molmod.io import dump_chk, load_chk, XYZWriter
import os

def get_modes(chk_file):

    chk = load_chk(chk_file)
    numbers = chk['numbers']
    hessian = chk['hessian'].reshape(3*len(numbers),3*len(numbers))
    masses = chk['masses']
    pos = chk['pos']
    gradient = chk['gradient']
    energy = chk['ene']

    mol = Molecule(numbers, pos, masses, energy, gradient, hessian)
    nma = NMA(mol)   

    return nma.freqs/lightspeed*centimeter, nma.modes.T, [nma.coordinates,nma.modes,nma.numbers,mol.masses3]

def transform_eig(modes,atoms,transform=np.eye(3)):

    atoms = np.concatenate((np.ravel(atoms[0]),np.ravel(atoms[1]),np.ravel(atoms[2])))
    modes_new = np.zeros(np.shape(modes))
    for i in xrange(len(modes)):	      
        for j in xrange(len(atoms)):
            modes_new[i,3*j:3*(j+1)] = np.dot([modes[i,3*atoms[j]],modes[i,3*atoms[j]+1],modes[i,3*atoms[j]+2]],transform) 

    return modes_new

def compare_modes(eig1,eig2=None):

    if eig2 is None:
        eig2 = eig1

    link_mode = []
    Correspondence = []
    Comparison = np.zeros((len(eig1),len(eig2)))
    for i in xrange(len(eig1)):
        max_value = 0
        mode = 0      
        for j in xrange(len(eig2)):
            value = np.dot(np.ravel(eig1[i]),np.ravel(eig2[j]))**2
            Comparison[i,j] = value
            if (max_value < value):
                max_value = value
                mode = j
        link_mode.append(mode)
        Correspondence.append(max_value)
    Correspondence = np.asarray(Correspondence)
    link_mode = np.asarray(link_mode)

    return link_mode, Correspondence, Comparison

def plot_matrix(matrix,fn,show=False):

    fig = pt.figure()
    ax = fig.add_subplot(111)

    pt.imshow(matrix[:50,:50],interpolation='nearest',cmap='jet')    
    pt.colorbar()
    pt.tick_params(
        axis='x',          
        which='both',      
        bottom='off',      
        top='off',
        labelbottom='off'
        )
    pt.tick_params(
        axis='y',          
        which='both',      
        right='off',      
        left='off',
        labelleft='off' 
        ) 
    pt.savefig(fn+'.png',bbox_inches='tight')
    if show:
        pt.show()        
    pt.close()

def assign_atoms(chk):

    sys = System.from_file(chk)
    num = sys.numbers
    metal = num[np.where(num>10)[0][0]]
    bond_dist = {(1,6):1.5*angstrom,(1,8):1.5*angstrom,(6,6):2.0*angstrom,(6,7):2.0*angstrom,(6,8):1.5*angstrom,(7,7):2.0*angstrom,(7,metal):2.5*angstrom,(8,metal):2.5*angstrom,(metal,metal):2.0*angstrom}
    sys.detect_bonds(exceptions=bond_dist)
    bonds = sys.bonds
    pos = sys.pos
    rvecs = sys.cell.rvecs

    #Find graph of the paddlewheels    
    pw_edges = []
    pw_criteria = {}    
    
    data = paddle_wheel()
    pw_edges += data[0]
    pw_criteria.update(data[1])
    data[1].clear()

    pw_atoms = make_graph(pw_edges, pw_criteria, bonds, num)
    
    #Find graph of the linkers
    linker_edges = []
    linker_criteria = {}    
    
    data = NDC()
    linker_edges += data[0]
    linker_criteria.update(data[1])
    data[1].clear()

    linker_atoms = make_graph(linker_edges, linker_criteria, bonds, num)

    #Find graph of the dabcos
    dabco_edges = []
    dabco_criteria = {}    
    
    data = dabco()
    dabco_edges += data[0]
    dabco_criteria.update(data[1])
    data[1].clear()

    dabco_atoms = make_graph(dabco_edges, dabco_criteria, bonds, num)
    
    return [pw_atoms,dabco_atoms,linker_atoms],sys

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

def NDC():
    
    graph_edges = [(0,1),(1,3),(3,7),(7,8),(8,5),(5,0),(8,12),(12,14),(14,15),(15,10),(10,7),(1,2),(3,4),(5,6),(9,10),(11,12),(13,14)]
    criteria = {0 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 6)), 1 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1)), 3 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1)), 7 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 6)), 8 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 6)), 5 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1)), 12 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1)), 14 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1)), 10 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 1)), 2 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 4 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 6 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 9 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 11 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 13 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 15: CritAnd(HasAtomNumber(6), HasNeighborNumbers(6, 6, 6))}
    
    return graph_edges, criteria

def paddle_wheel():

    graph_edges = [(0,2),(0,3),(0,4),(0,5),(1,6),(1,7),(1,8),(1,9),(2,10),(6,10),(3,11),(7,11),(4,12),(8,12),(5,13),(9,13)]
    criteria = {0 : HasNeighborNumbers(8,8,8,8,7), 1 : HasNeighborNumbers(8,8,8,8,7), 2 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 3 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 4 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 5 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 6 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 7 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 8 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 9 : CritAnd(HasAtomNumber(8), HasNumNeighbors(2)), 10: CritAnd(HasAtomNumber(6), HasNeighborNumbers(8, 8, 6)), 11: CritAnd(HasAtomNumber(6), HasNeighborNumbers(8, 8, 6)), 12: CritAnd(HasAtomNumber(6), HasNeighborNumbers(8, 8, 6)), 13: CritAnd(HasAtomNumber(6), HasNeighborNumbers(8, 8, 6))}

    return graph_edges, criteria

def dabco():
    
    graph_edges = [(0,2),(0,3),(0,4),(1,5),(1,6),(1,7),(2,8),(2,9),(3,10),(3,11),(4,12),(4,13),(5,14),(5,15),(6,16),(6,17),(7,18),(7,19),(2,5),(3,6),(4,7)]
    criteria = {0 : CritAnd(HasAtomNumber(7), HasNumNeighbors(4)), 1 : CritAnd(HasAtomNumber(7), HasNumNeighbors(4)), 2 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 1, 1)), 3 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 1, 1)), 4 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 1, 1)), 5 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 1, 1)), 6 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 1, 1)), 7 : CritAnd(HasAtomNumber(6), HasNeighborNumbers(7, 6, 1, 1)), 8 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 9 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 10 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 11 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 12 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 13 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 14 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 15 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 16 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 17 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 18 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6)), 19 : CritAnd(HasAtomNumber(1), HasNeighborNumbers(6))}

    return graph_edges, criteria

def com_distance(atoms,system):

    distance = np.zeros((len(system.numbers),3))
    com = np.zeros(3)
    ref_pos = system.pos[atoms[0],:]
    for j in atoms:
        vec = system.pos[j,:]-ref_pos
        system.cell.mic(vec)
        com += (ref_pos+vec)*system.masses[j]/np.sum(system.masses[atoms])
    for j in atoms:
        vec = system.pos[j,:]-com
        system.cell.mic(vec)
        distance[j,:] = vec.copy()

    return distance 

def rotation(atoms,system,rot_axis):
   
    distance = com_distance(atoms,system)
    rot = np.zeros((len(system.numbers),3))
    for j in atoms:
        rot[j,:] = np.cross(rot_axis,distance[j,:])*np.sqrt(system.masses[j])
    
    return np.ravel(rot)/np.linalg.norm(np.ravel(rot))

def translation(atoms,system,direction):
    
    trans = np.zeros((len(system.numbers),3))
    for j in atoms:
        trans[j,:] += direction*np.sqrt(system.masses[j])    

    return np.ravel(trans)/np.linalg.norm(np.ravel(trans))

def mom_inert(atoms,system):

    tensor = np.zeros((3,3))
    dist = com_distance(atoms,system)
    for j in atoms:
        tensor += system.masses[j]*(np.linalg.norm(dist[j])**2*np.eye(3)-np.einsum('i,j->ij',dist[j],dist[j]))

    eigval,eigvec=np.linalg.eigh(tensor)

    return eigvec

def get_contributions(modes,atoms,system):

    trans_rot = np.zeros((8*6,len(modes)))
    counter = 0
    for i in range(len(atoms)):
        for j in range(len(atoms[i])):
            inert = mom_inert(atoms[i][j],system)
            trans_rot[6*counter+0,:] = translation(atoms[i][j],system,inert[:,0])
            trans_rot[6*counter+1,:] = translation(atoms[i][j],system,inert[:,1])
            trans_rot[6*counter+2,:] = translation(atoms[i][j],system,inert[:,2])
            trans_rot[6*counter+3,:] = rotation(atoms[i][j],system,inert[:,0])
            trans_rot[6*counter+4,:] = rotation(atoms[i][j],system,inert[:,1])
            trans_rot[6*counter+5,:] = rotation(atoms[i][j],system,inert[:,2])
            counter += 1
    
    contributions = np.einsum('ik,jk->ij',modes,trans_rot)       

    return contributions

def mode_character(contr):
    
    character = np.zeros((len(contr),7))
    for i in range(len(contr)):
        character[i,0] = np.sum(contr[i,0:3]**2)+np.sum(contr[i,6:9]**2) #pw trans
        character[i,1] = np.sum(contr[i,3:6]**2)+np.sum(contr[i,9:12]**2) #pw rot
        character[i,2] = np.sum(contr[i,12:15]**2)+np.sum(contr[i,18:21]**2) #dabco trans
        character[i,3] = np.sum(contr[i,15:18]**2)+np.sum(contr[i,21:24]**2) #dabco rot  
        character[i,4] = np.sum(contr[i,24:27]**2)+np.sum(contr[i,30:33]**2)+np.sum(contr[i,36:39]**2)+np.sum(contr[i,42:45]**2) #linker trans
        character[i,5] = np.sum(contr[i,27:30]**2)+np.sum(contr[i,33:36]**2)+np.sum(contr[i,39:42]**2)+np.sum(contr[i,45:48]**2) #linker rot    
        character[i,6] = 1-np.sum(character[i,0:6]) #other

    return character  

def plot_character(ax, modes, character):

    colors = ['#E31A1C','#FD8D3C','#238445','#78C679','#0570B0','#A6BDDB','#d9d9d9']

    pos = np.arange(len(modes))   
    ax.bar(pos, character[modes,6], bottom=np.sum(character[modes,:6],axis=1),width=0.7, color=colors[6], label='other')  
    ax.bar(pos, character[modes,5], bottom=np.sum(character[modes,:5],axis=1),width=0.7, color=colors[5], label='linker rotation') 
    ax.bar(pos, character[modes,4], bottom=np.sum(character[modes,:4],axis=1),width=0.7, color=colors[4], label='linker translation')
    ax.bar(pos, character[modes,3], bottom=np.sum(character[modes,:3],axis=1),width=0.7, color=colors[3], label='dabco rotation')
    ax.bar(pos, character[modes,2], bottom=np.sum(character[modes,:2],axis=1),width=0.7, color=colors[2], label='dabco translation') 
    ax.bar(pos, character[modes,1], bottom=np.sum(character[modes,:1],axis=1),width=0.7, color=colors[1], label='pw rotation')  
    ax.bar(pos, character[modes,0], width=0.7, color=colors[0], label='pw translation')      
    ax.legend(bbox_to_anchor=(1.15, 0.63))
    ax.set_xticks([])
    ax.set_xlim(-1,len(modes))
    ax.set_yticks([0.0,1.0])
    ax.set_ylim(0,1.0)
    ax.set_xlabel('Terahertz modes',fontsize=16)
    ax.set_ylabel(r'Mode character',fontsize=16)

def plot_freq(ax, modes, freqs):

    for i in range(len(modes)/2):
        ax.axvspan(2*i-0.5,2*i+0.5,facecolor='lightgray',alpha=0.5)

    pos = np.arange(len(modes)) 
    ax.grid(True,axis='y')
    ax.plot(pos,freqs[modes],marker='o',linestyle='')
    ax.set_xticks([])
    ax.set_xlim(-1,len(modes))
    ax.set_ylim(-10,135)
    ax.set_yticks([0,25,50,75,100,125])
    ax.set_ylabel(r'Frequency (cm$^{-1}$)',fontsize=16)

def plot_freq_all(ax, modes, link, freqs, structures):

    for i in range(len(modes)/2):
        ax.axvspan(2*i-0.5,2*i+0.5,facecolor='lightgray',alpha=0.5)

    colors = ['#3288bd','#abdda4','#fdae61','#d53e4f']

    pos = np.arange(len(modes)) 
    ax.grid(True,axis='y')
    for i in range(len(structures)):
        ax.plot(pos,freqs[i][link[i][modes]],marker='o',linestyle='',color=colors[i],label=structures[i].split('_')[1])
    ax.legend(bbox_to_anchor=(1.15, 0.6))
    ax.set_xticks([])
    ax.set_xlim(-1,len(modes))
    ax.set_ylim(-10,135)
    ax.set_yticks([0,25,50,75,100,125])
    ax.set_ylabel(r'Frequency (cm$^{-1}$)',fontsize=16)

def plot_freq_diff(ax, modes, link, corr, freqs, structures):     

    for i in range(len(structures)):
        structures[i] = structures[i].split('_')[1] 

    #Get frequency difference between structures
    freqs_ref = freqs[np.where(structures=='Zn')][0]
    diff = np.zeros((len(link),len(modes)))
    for i in range(len(link)):
        for j in range(len(modes)):
            if corr[i][modes[j]] > 0.25:
                diff[i,j] = freqs[i][link[i][modes[j]]] - freqs_ref[modes[j]]
            else:
                diff[i,j] = 0

    colors = ['#4a1486','#807dba','#bcbddc']

    #Plot difference
    for i in range(len(modes)/2):
        ax.axvspan(2*i-0.5,2*i+0.5,facecolor='lightgray',alpha=0.5)

    pos = np.arange(len(modes)) 
    width = np.linspace(-0.25,0.5,num=len(structures))
    for j in range(len(structures)-1):      
        ax.bar(pos+width[j], diff[j,:], width=width[j+1]-width[j], color=colors[j], label=structures[j],zorder=3)
    ax.grid(True,axis='y',zorder=0)
    ax.legend(bbox_to_anchor=(1.15, 0.56))
    ax.set_xticks([])
    ax.set_xlim(-1,len(modes))
    ax.set_ylim(-50,50)
    ax.set_ylabel(r'Frequency difference (cm$^{-1}$)',fontsize=16)

def plot_lp_cp(ref_link,ref_corr,lp_link,lp_corr,cp_link,cp_corr,freqs_lp,freqs_cp,structures):

    modes = np.arange(50)      

    for i in range(len(structures)):
        structures[i] = structures[i].split('_')[1]

    #Get difference between lp structures
    freqs_lp_ref = freqs_lp[np.where(structures=='Zn')][0]
    diff_lp = np.zeros((len(lp_link),len(modes)))
    for i in range(len(lp_link)):
        for j in range(len(modes)):
            if lp_corr[i][modes[j]] > 0.25:
                diff_lp[i,j] = freqs_lp[i][lp_link[i][modes[j]]] - freqs_lp_ref[modes[j]]
            else:
                diff_lp[i,j] = 1000
    
    print structures[0], structures[1], structures[2], structures[3]
    for j in range(len(modes)):
        print modes[j], freqs_lp[0][lp_link[0][modes[j]]], freqs_lp[1][lp_link[1][modes[j]]], freqs_lp[2][lp_link[2][modes[j]]], freqs_lp[3][lp_link[3][modes[j]]]
    
    #Get difference between cp structures
    freqs_cp_ref = freqs_cp[np.where(structures=='Zn')][0]
    diff_cp = np.zeros((len(cp_link),len(modes)))
    for i in range(len(cp_link)):
        for j in range(len(modes)):
            if cp_corr[i][ref_link[modes[j]]] > 0.25:
                diff_cp[i,j] = freqs_cp[i][cp_link[i][ref_link[modes[j]]]] - freqs_cp_ref[ref_link[modes[j]]]
            else:
                diff_cp[i,j] = 2000

    #Get difference between phases
    diff_cp_lp = diff_cp - diff_lp
    for i in range(len(modes)):
        diff_cp_lp[:,i] += freqs_cp_ref[ref_link[modes[i]]] - freqs_lp_ref[modes[i]]

    colors = ['#d7191c','#fdae61','#abdda4','#2b83ba']

    #Plot difference lp
    fig = pt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    pos = 0.0
    width = np.linspace(0,1.0,num=len(structures)+1)
    for i in range(len(modes)):
        for j in range(len(structures)-1):      
            if np.abs(diff_lp[j,i])<200:
                ax.bar(pos+width[j], diff_lp[j,i], width=width[j+1]-width[j], color=colors[j], label=structures[j] if i == 0 else "")
        pos += 1
       
    pt.legend(loc=3)
    pt.xticks([])
    pt.ylim(-60,60)
    pt.xlabel('Terahertz modes',fontsize=16)
    pt.ylabel(r'Frequency difference (cm$^{-1}$)',fontsize=16)
    pt.savefig('Relation_terahertz_modes_lp.pdf',transparent='true',bbox_inches='tight')
    pt.close()
    
    #Plot difference cp     
    fig = pt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    pos = 0.0
    width = np.linspace(0,1.0,num=len(structures)+1)
    for i in range(len(modes)):
        for j in range(len(structures)-1):      
            if np.abs(diff_cp[j,i])<200:
                ax.bar(pos+width[j], diff_cp[j,i], width=width[j+1]-width[j], color=colors[j], label=structures[j] if i == 0 else "")
        pos += 1
       
    pt.legend(loc=3)
    pt.xticks([])
    pt.ylim(-60,60)
    pt.xlabel('Terahertz modes',fontsize=16)
    pt.ylabel(r'Frequency difference (cm$^{-1}$)',fontsize=16)
    pt.savefig('Relation_terahertz_modes_cp.pdf',transparent='true',bbox_inches='tight')
    pt.close()

    #Plot difference cp-lp  
    fig = pt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    pos = 0.0
    width = np.linspace(0,1.0,num=len(structures)+2)
    for i in range(len(modes)):
        for j in range(len(structures)):      
            if np.abs(diff_cp_lp[j,i])<200:
                ax.bar(pos+width[j], diff_cp_lp[j,i], width=width[j+1]-width[j], color=colors[j], label=structures[j] if i == 0 else "")
        pos += 1
       
    pt.legend(loc=3)
    pt.xticks([])
    pt.ylim(-100,100)
    pt.xlabel('Terahertz modes',fontsize=16)
    pt.ylabel(r'Frequency difference (cm$^{-1}$)',fontsize=16)
    pt.savefig('Relation_terahertz_modes_cp_lp.pdf',transparent='true',bbox_inches='tight')
    pt.close()

if __name__ == '__main__':
    
    path = "chk"
    files = os.listdir(path)
    files = np.sort(files)

    mode_data = []
    mode_freqs = []
    character = []
    for fn in files:
        atoms,system = assign_atoms(path+'/'+fn) 
        freqs, modes, data = get_modes(path+'/'+fn)
        contr = get_contributions(modes,atoms,system) 
        mode_data.append(modes)
        mode_freqs.append(freqs)
        character.append(mode_character(contr))
        if fn=='DUT8_Zn_lp.chk':
            contr = get_contributions(modes,atoms,system) 
            character_lp = mode_character(contr)
            for i in range(len(freqs)):
                print freqs[i]
        elif fn=='DUT8_Zn_cp.chk':
            contr = get_contributions(modes,atoms,system) 
            character_cp = mode_character(contr)
    mode_freqs = np.asarray(mode_freqs)

    for i in range(len(files)):
        files[i] = files[i].split('.')[0]
    files = files.tolist()

    ref_lp = files.index('DUT8_Zn_lp')
    ref_cp = files.index('DUT8_Zn_cp')
    modes_ref_link, modes_ref_corr, ref_lp_cp = compare_modes(mode_data[ref_lp],mode_data[ref_cp])
    plot_matrix(ref_lp_cp,'plots/lp-cp')

    matrix_lp = []
    files_lp = []
    modes_lp_link = []
    modes_lp_corr = []
    matrix_cp = []
    files_cp = []
    modes_cp_link = []
    modes_cp_corr = []
    for i in range(len(mode_data)):
        if 'lp' in files[i]:
            link_lp, corr_lp, comp_lp = compare_modes(mode_data[ref_lp],mode_data[i])
            matrix_lp.append(comp_lp)
            modes_lp_link.append(link_lp)
            modes_lp_corr.append(corr_lp)
            plot_matrix(matrix_lp[-1],'plots/'+files[i])
            files_lp.append(i)
        elif 'cp' in files[i]:
            link_cp, corr_cp, comp_cp = compare_modes(mode_data[ref_cp],mode_data[i])
            matrix_cp.append(comp_cp)
            modes_cp_link.append(link_cp)
            modes_cp_corr.append(corr_cp)
            plot_matrix(matrix_cp[-1],'plots/'+files[i])
            files_cp.append(i)

    files_lp = np.asarray(files_lp,dtype=np.int)
    files_cp = np.asarray(files_cp,dtype=np.int)
    files = np.asarray(files)

    THz_modes = np.arange(30) 
    
    pt.close('all')
    fig = pt.figure(figsize=(16,20),tight_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1,2,2])
    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[1,0])
    ax3 = fig.add_subplot(spec[2,0])
    plot_freq(ax1,THz_modes,mode_freqs[ref_lp])
    plot_freq_diff(ax2,THz_modes,modes_lp_link,modes_lp_corr,mode_freqs[files_lp],files[files_lp])
    plot_character(ax3,THz_modes,character_lp)
    pt.savefig('DUT8_lp.pdf',transparent='true',bbox_inches='tight')

    pt.close('all')
    fig = pt.figure(figsize=(16,20),tight_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1,2,2])
    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[1,0])
    ax3 = fig.add_subplot(spec[2,0])
    plot_freq(ax1,THz_modes,mode_freqs[ref_cp])
    plot_freq_diff(ax2,THz_modes,modes_cp_link,modes_cp_corr,mode_freqs[files_cp],files[files_cp])
    plot_character(ax3,THz_modes,character_cp)
    pt.savefig('DUT8_cp.pdf',transparent='true',bbox_inches='tight')
    
    pt.close('all')
    fig = pt.figure(figsize=(16,20),tight_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=5)
    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[1,0])
    ax3 = fig.add_subplot(spec[2,0])
    ax4 = fig.add_subplot(spec[3,0])
    ax5 = fig.add_subplot(spec[4,0])
    plot_freq_all(ax1,THz_modes,modes_lp_link,mode_freqs[files_lp],files[files_lp])
    for count,i in enumerate(files_lp):
        if files[i].split('_')[1]=='Zn':
            plot_character(ax2,THz_modes,character[i][modes_lp_link[count][THz_modes]])
        elif files[i].split('_')[1]=='Co':
            plot_character(ax3,THz_modes,character[i][modes_lp_link[count][THz_modes]])
        elif files[i].split('_')[1]=='Cu':
            plot_character(ax4,THz_modes,character[i][modes_lp_link[count][THz_modes]])
        elif files[i].split('_')[1]=='Ni':
            plot_character(ax5,THz_modes,character[i][modes_lp_link[count][THz_modes]])
    pt.savefig('DUT8_lp_character.pdf',transparent='true',bbox_inches='tight')

    pt.close('all')
    fig = pt.figure(figsize=(16,20),tight_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=5)
    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[1,0])
    ax3 = fig.add_subplot(spec[2,0])
    ax4 = fig.add_subplot(spec[3,0])
    ax5 = fig.add_subplot(spec[4,0])
    plot_freq_all(ax1,THz_modes,modes_cp_link,mode_freqs[files_cp],files[files_cp])
    for count,i in enumerate(files_cp):
        if files[i].split('_')[1]=='Zn':
            plot_character(ax2,THz_modes,character[i][modes_cp_link[count][THz_modes]])
        elif files[i].split('_')[1]=='Co':
            plot_character(ax3,THz_modes,character[i][modes_cp_link[count][THz_modes]])
        elif files[i].split('_')[1]=='Cu':
            plot_character(ax4,THz_modes,character[i][modes_cp_link[count][THz_modes]])
        elif files[i].split('_')[1]=='Ni':
            plot_character(ax5,THz_modes,character[i][modes_cp_link[count][THz_modes]])
    pt.savefig('DUT8_cp_character.pdf',transparent='true',bbox_inches='tight')
    

