import numpy as np

"""
Nerual net format:
    nnet["numLayers"] = number of layers in the nnet
    nnet["inputSize"] = number of input neurons
    nnet["outputSize"] = num output neurons
    nnet["maxLayerSize"] = max num neurons in any layer
    nnet["isSymmetric"] = not used
    nnet["minVals"] = list of min values for each input neuron
    nnet["maxVals"] = list of max values for each input neuron
    nnet["means"] = list of means for each input
    nnet["ranges"] = list of ranges for each input
    nnet["weights"] = 3D list
    nnet["biases"] = 2D list

"""

def read_network(nnet_path):
    ret = {}
    strip_chars = ", \n"
    with open(nnet_path, "r", encoding="utf-8") as nnet_fd:
        tmp = nnet_fd.readline()
        while tmp.startswith("/"):
            tmp = nnet_fd.readline()

        params = list(map(int, tmp.strip(strip_chars).split(",")))
        ret["numLayers"] = params[0]
        ret["inputSize"] = params[1]
        ret["outputSize"] = params[2]
        ret["maxLayerSize"] = params[3]

        ret["layerSizes"] = list(map(int, nnet_fd.readline().strip(strip_chars).split(",")))

        ret["isSymmetric"] = list(map(int, nnet_fd.readline().strip(strip_chars).split(",")))[0]

        ret["minVals"] = nnet_fd.readline().strip(strip_chars).split(",")
        ret["maxVals"] = nnet_fd.readline().strip(strip_chars).split(",")
        ret["means"] = nnet_fd.readline().strip(strip_chars).split(",")
        ret["ranges"] = nnet_fd.readline().strip(strip_chars).split(",")


        weights = []
        biases = []
        for i in range(0, ret["numLayers"]):
            matrix = []
            bias = []
            for j in range(0, ret["layerSizes"][i + 1]):
                matrix.append(nnet_fd.readline().strip(strip_chars).split(","))
            for _ in range(0, ret["layerSizes"][i + 1]):
                bias.append(nnet_fd.readline().strip(strip_chars))
            weights.append(matrix)
            biases.append(bias)
        ret["weights"] = weights
        ret["biases"] = biases

    return ret


def write_network(nnet, out_path):
    with open(out_path, "w+", encoding="utf-8") as out_fd:
        out_fd.write(
            ",".join(map(str,
                [nnet["numLayers"], nnet["inputSize"], nnet["outputSize"], nnet["maxLayerSize"]]
            )) + ",\n")
        out_fd.write(",".join(map(str, nnet["layerSizes"])) + ",\n")
        out_fd.write(str(nnet["isSymmetric"]) + ",\n")
        out_fd.write(",".join(map(str, nnet["minVals"])) + ",\n")
        out_fd.write(",".join(map(str, nnet["maxVals"])) + ",\n")
        out_fd.write(",".join(map(str, nnet["means"])) + ",\n")
        out_fd.write(",".join(map(str, nnet["ranges"])) + ",\n")

        for i in range(0, nnet["numLayers"]):
            for j in range(0, nnet["layerSizes"][i + 1]):
                out_fd.write(",".join(map(lambda x: str(np.float32(x)) if type(x) is float else str(x), nnet["weights"][i][j])) + ",\n")
            out_fd.write(",\n".join(map(lambda x: str(np.float32(x)) if type(x) is float else str(x), nnet["biases"][i])) + ",\n")
