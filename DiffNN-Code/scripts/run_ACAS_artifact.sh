#!/bin/bash

epsilon="0.05"

./delta_network_test 16 nnet/ACASXU_run2a_1_1_batch_2000.nnet compressed_nnets/ACASXU_run2a_1_1_batch_2000_16bit.nnet $epsilon
./delta_network_test 26 nnet/ACASXU_run2a_1_1_batch_2000.nnet compressed_nnets/ACASXU_run2a_1_1_batch_2000_16bit.nnet $epsilon

