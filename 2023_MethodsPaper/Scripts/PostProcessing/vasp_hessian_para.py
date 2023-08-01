#!/usr/bin/env python

#Generation of POSCAR files with selective dynamics to split up Hessian calculation

natom = 456  # Number of atoms in the unit cell

import os

for i in range(natom):
    os.mkdir('%s'%i)
    old = open('POSCAR','r')
    new = open('%s/POSCAR'%i,'w')
    n = 0
    for line in old:
        if n==7: new.write('Selective\n%s'%line)
        elif n==(8+i): new.write(line[:-1] + ' T T T\n')
        elif n>7: new.write(line[:-1] + ' F F F\n')
        else: new.write(line)
        n+=1
    old.close()
    new.close()
    os.system('cp INCAR %s'%i)
    os.system('cp POTCAR %s'%i)
    os.system('cp KPOINTS %s'%i)

