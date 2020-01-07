#!/bin/bash

epsilon="0.05"

./network_test 16 subbed_nnets/ACASXU_run2a_1_1_batch_2000_16bit.nnet $epsilon
./network_test 26 subbed_nnets/ACASXU_run2a_1_1_batch_2000_16bit.nnet $epsilon

