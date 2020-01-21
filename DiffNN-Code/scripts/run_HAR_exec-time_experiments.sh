#!/bin/bash
perturb=0.1
eps=0.25
TIMEOUT=600
out=exec-time_out_HAR
> $out

for testcase in $( seq 1000 1099 ); do
	echo "./delta_network_test $testcase nnet/HAR.nnet compressed_nnets/HAR_16bit.nnet $eps -p $perturb" >> $out
	timeout $TIMEOUT ./delta_network_test $testcase nnet/HAR.nnet compressed_nnets/HAR_16bit.nnet $eps -p $perturb 2>> $out
done
