from common import *
import sys

def round_nnet_16bit(nnet):
    import numpy as np

    weights = nnet["weights"]
    biases = nnet["biases"]
    new_weights = []
    new_biases = []
    for k, weightMatrix in enumerate(weights):
        matrix = []
        for row in weightMatrix:
            matrix.append(list(map(lambda f: np.float16(f), row)))
        new_weights.append(matrix)

        bias = list(map(lambda f: np.float16(f), biases[k]))
        new_biases.append(bias)
    nnet["weights"] = new_weights
    nnet["biases"] = new_biases
    return nnet

def main():
    if len(sys.argv) != 3:
        print("usage: python %s <nnet-path> <output-path> " % (sys.argv[0]))
        exit(1)

    nnet = read_network(sys.argv[1])

    rounded_nnet = round_nnet_16bit(nnet)

    write_network(rounded_nnet, sys.argv[2])

if __name__ == "__main__":
    main()
