import sys
from common import *


def subtract_nnets(nnet1, nnet2, noAffineMode=False):
    if nnet2["layerSizes"] != nnet1["layerSizes"]:
        print("Networks have different structures")
        exit(1)

    new_net = {}

    # add one layer to subtract the outputs
    new_net["numLayers"] = nnet1["numLayers"] + 1
    new_net["inputSize"] = nnet1["inputSize"]
    new_net["outputSize"] = nnet1["outputSize"]
    new_net["maxLayerSize"] = 2*nnet1["maxLayerSize"]


    new_net["layerSizes"] = [nnet1["inputSize"]]
    for i in range(1, len(nnet1["layerSizes"])):
        new_net["layerSizes"].append(nnet1["layerSizes"][i]*2)
    new_net["layerSizes"].append(nnet1["outputSize"])

    new_net["isSymmetric"] = nnet1["isSymmetric"]

    new_net["minVals"] = list(nnet1["minVals"])
    new_net["maxVals"] = list(nnet1["maxVals"])
    new_net["means"] = list(nnet1["means"])
    new_net["ranges"] = list(nnet1["ranges"])

    weights = []
    biases = []

    # first layer is a special case
    matrix = []
    for row in nnet1["weights"][0]:
        matrix.append(list(row))
    for row in nnet2["weights"][0]:
        matrix.append(list(row))
    weights.append(matrix)
    biases.append(list(nnet1["biases"][0]) + list(nnet2["biases"][0]))

    for i in range(1, len(nnet1["weights"])):
        matrix = []
        for j in range(0, nnet1["layerSizes"][i+1]):
            matrix.append(list(nnet1["weights"][i][j]) + len(nnet1["weights"][i][j])*["0.0"])
        for j in range(0, nnet1["layerSizes"][i+1]):
            matrix.append(len(nnet1["weights"][i][j])*["0.0"] + list(nnet2["weights"][i][j]))
        weights.append(matrix)
        if noAffineMode and i == len(nnet1["weights"]) - 1:
            biases.append(map(lambda b: str(float(b) + 100.0), list(nnet1["biases"][i]) + list(nnet2["biases"][i])))
        else:
            biases.append(list(nnet1["biases"][i]) + list(nnet2["biases"][i]))


    # now make the new output layer that subtracts the parameters
    matrix = []
    for i in range(0, nnet1["outputSize"]):
        row = []
        if i > 0:
            row.extend(["0.0"]*i)
        row.append("1.0")
        if i < nnet1["outputSize"] - 1:
            row.extend(["0.0"]*(nnet1["outputSize"] - 1 - i))
        row1 = list(row)
        row1[row.index("1.0")] = "-1.0"
        row.extend(row1)
        matrix.append(row)
    weights.append(matrix)
    biases.append(["0.0"]*nnet1["outputSize"])

    new_net["weights"] = weights
    new_net["biases"] = biases

    return new_net

def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("usage: python " + sys.argv[0] + " <nnet1-path> <nnet2-path> <out.nnet>")
        exit(1)
    
    nnet1 = read_network(sys.argv[1])
    nnet2 = read_network(sys.argv[2])


    res = subtract_nnets(nnet1, nnet2, False)

    write_network(res, sys.argv[3])

if __name__ == "__main__":
    main()
