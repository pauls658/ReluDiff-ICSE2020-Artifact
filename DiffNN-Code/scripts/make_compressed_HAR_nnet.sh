#!/bin/bash

nnet="HAR.nnet"

python3 python/round_nnet.py nnet/$nnet \
		compressed_nnets/${nnet/\.nnet/_16bit.nnet}
python3 python/subtract_nnets.py nnet/$nnet \
		compressed_nnets/${nnet/\.nnet/_16bit.nnet} \
		../ReluVal-for-comparison/subbed_nnets/${nnet/\.nnet/_16bit.nnet}
