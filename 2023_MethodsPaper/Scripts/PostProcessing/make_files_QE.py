import numpy as np
from molmod.units import *
from molmod.periodic import *
from QEinput import *
import os
import shutil

fn = 'input'
QE = QEModel('POSCAR',prefix='UiO66',cutoff=44,kpoints=(1,1,1))
if not os.path.exists('p000'):  
    os.mkdir('p000')
QE.write_pw_input('p000/'+fn,efield=np.array([0,0,0]))
for i in ['100','010','001','110','101','011']:
    for j in [-2,-1,1,2]:
        index = 'p'
        if j<0:
            index = 'm'
        field = np.array([int(i[0]),int(i[1]),int(i[2])])*j
        if not os.path.exists('{}{}{}{}/'.format(index,*np.abs(field))):  
            os.mkdir('{}{}{}{}/'.format(index,*np.abs(field)))
        QE.write_pw_input('{}{}{}{}/'.format(index,*np.abs(field))+fn,efield=field/np.linalg.norm(field)*0.001*np.abs(j))


