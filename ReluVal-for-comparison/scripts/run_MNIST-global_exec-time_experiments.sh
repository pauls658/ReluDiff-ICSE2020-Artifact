#!/bin/bash

perturb=3
eps=1
timeout=1800
subbed_nnets="subbed_nnets/mnist_relu_2_512_16bit.nnet
subbed_nnets/mnist_relu_3_100_16bit.nnet
subbed_nnets/mnist_relu_4_1024_16bit.nnet"

IFS=$'\n'
for subbed_nnet in $subbed_nnets; do
	out="exec-time_out_${subbed_nnet/subbed_nnets\//}"
	> $out
	for testcase in $( seq 400 499 ); do
		orig_nnet=${subbed_nnet/subbed_nnets/nnet}
		orig_nnet=${orig_nnet/\_16bit\.nnet/\.nnet}
		echo "./network_test $testcase $subbed_nnet $eps $perturb" >> $out
		timeout -s2 $timeout ./network_test \
			$testcase \
			$subbed_nnet\
			$eps -p $perturb 2>> $out
	done
done
