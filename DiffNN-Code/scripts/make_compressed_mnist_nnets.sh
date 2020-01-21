#!/bin/bash

mkdir -p compressed_nnets
mkdir -p ../ReluVal-for-comparison/subbed_nnets

nnets="mnist_relu_4_1024.nnet
mnist_relu_3_100.nnet
mnist_relu_2_512.nnet"

for nnet in $nnets; do
	python3 python/round_nnet.py nnet/$nnet \
			compressed_nnets/${nnet/\.nnet/_16bit.nnet}
	python3 python/subtract_nnets.py nnet/$nnet \
			compressed_nnets/${nnet/\.nnet/_16bit.nnet} \
			../ReluVal-for-comparison/subbed_nnets/${nnet/\.nnet/_16bit.nnet}
done
