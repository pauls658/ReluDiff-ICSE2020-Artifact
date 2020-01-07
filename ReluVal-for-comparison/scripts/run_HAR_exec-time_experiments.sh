#!/bin/bash
perturb=0.1
eps=0.25
TIMEOUT=1800
out=HAR_exec-time_out
> $out

for testcase in $( seq 1000 1099 ); do
	echo "./network_test $testcase subbed_nnets/HAR_16bit.nnet $eps $perturb -1" >> $out
	timeout $TIMEOUT ./network_test $testcase subbed_nnets/HAR_16bit.nnet $eps $perturb -1 2>> $out
done
