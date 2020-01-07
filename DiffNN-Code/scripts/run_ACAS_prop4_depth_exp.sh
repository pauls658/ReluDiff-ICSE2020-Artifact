#!/bin/bash

make clean depth bench v1 all
out=acas_prop4_depths
> $out
for compressed_nnet in $( ls compressed_nnets/ACAS* ); do
	orig_nnet=${compressed_nnet/_16bit/}
	orig_nnet=${orig_nnet/compressed_nnets/nnet}
	./delta_network_test 4 $orig_nnet $compressed_nnet 0.05 2>> $out
done
