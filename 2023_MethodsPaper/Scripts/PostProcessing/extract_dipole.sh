#!/bin/bash
#Extract dipole moments from MD run in CP2K

sed -n '6,8p' E0/run1/moments.dat > cell_tmp.txt
grep '\[X\]' cell_tmp.txt | awk '{print $3,'t',$4,'t',$5}' > cellx.txt
grep '\[Y\]' cell_tmp.txt | awk '{print $2,'t',$3,'t',$4}' > celly.txt
grep '\[Z\]' cell_tmp.txt | awk '{print $3,'t',$4,'t',$5}' > cellz.txt
cat cellx.txt celly.txt cellz.txt > cell.txt
rm cellx.txt celly.txt cellz.txt cell_tmp.txt
for j in {E0,Ex,Ey,Ez};
    do
    > $j/dipole.txt
    for k in {run1,run2,run3,run4,run5,run6,run7,run8};
        do
        if [ -f $j/$k/moments.dat ]; then 
            grep 'X=' $j/$k/moments.dat | awk '{print $2,'t',$4,'t',$6}' > $j/$k/moments.txt 
            sed -n '2,1251p' $j/$k/moments.txt > $j/$k/moments_cut.txt

            cat $j/dipole.txt $j/$k/moments_cut.txt > $j/temp.txt; mv $j/temp.txt $j/dipole.txt
            wc -l < $j/dipole.txt
        #Check if all simulations are finished properly
        #else
        #    echo $j/$k
        fi
    done
    if [ -f $j/dipole.txt ]; then 
        echo $j
        wc -l < $j/dipole.txt
    fi
done

