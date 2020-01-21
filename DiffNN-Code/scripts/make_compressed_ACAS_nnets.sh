#!/bin/bash

mkdir -p compressed_nnets
mkdir -p ../ReluVal-for-comparison/subbed_nnets

for nnet in nnet/ACAS*; do
	compressed_nnet=${nnet/nnet/compressed_nnets}
	compressed_nnet=${compressed_nnet/\.nnet/\_16bit.nnet}
	python3 python/round_nnet.py $nnet $compressed_nnet
	python3 python/subtract_nnets.py $nnet $compressed_nnet \
			${compressed_nnet/compressed_nnets/..\/ReluVal-for-comparison/subbed_nnets}
done
