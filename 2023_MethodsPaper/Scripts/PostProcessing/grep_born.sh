# extract Born effective charges tensors
BORN_NROWS=`grep NIONS OUTCAR | awk '{print $12*4+1}'`
if [ `grep 'BORN' OUTCAR | wc -l` = 0 ] ; then \
   printf " .. FAILED! Born effective charges missing! Bye! \n\n" ; exit 1 ; fi
grep "in |e|, cummulative" -A $BORN_NROWS OUTCAR > born.txt

