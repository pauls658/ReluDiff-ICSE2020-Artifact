#!/bin/bash

make clean depth bench all

out=acas_prop4_depths
> $out
for compressed_nnet in $( ls subbed_nnets/ACAS* ); do
	timeout -s 2 1800 ./network_test 4 $compressed_nnet 0.05 2>> $out
done
