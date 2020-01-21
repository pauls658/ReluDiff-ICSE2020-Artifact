import sys, csv, time, re
from collections import OrderedDict
sys.path.insert(0, '../ELINA/python_interface/')
from read_net_file import *
from eran import ERAN
from analyzer import *

import numpy as np

nnetPrefix = "../NNet/nnet/"

"""
Does the same thing as getEps, but using subtract output neurons.
Requires that the nnet does not have the final "subtraction layer".
"""
def getEps2(eran, specLB, specUB, target, numOutputs):
    specLB = np.reshape(specLB, (-1,))
    specUB = np.reshape(specUB, (-1,))
    nn = layers()
    nn.specLB = specLB
    nn.specUB = specUB
    execute_list = eran.optimizer.get_deeppoly(specLB, specUB)

    start = time.clock()
    analyzer = Analyzer(execute_list, nn, "deeppoly", 0, 0, 0, True)
    element, lb, ub = analyzer.get_abstract0()
    res = subtract_output_neurons(analyzer.man, element, target, target + numOutputs, True)
    elapsed = time.clock() - start
    elina_abstract0_free(analyzer.man, element)
    return (elapsed, res.contents.sup.contents.val.dbl, res.contents.inf.contents.val.dbl)

prop16UB = [0.679857730865478515625000000000, -0.111408457159996032714843750000, -0.499204099178314208984375000000, -0.409090906381607055664062500000, 0.500000000000000000000000000000, ]
prop16LB = [-0.129289120435714721679687500000, -0.499999880790710449218750000000, -0.499999880790710449218750000000, -0.500000000000000000000000000000, -0.500000000000000000000000000000, ]
prop26UB = [0.679857730865478515625000000000, 0.499999880790710449218750000000, -0.499204099178314208984375000000, -0.409090906381607055664062500000, 0.500000000000000000000000000000, ]
prop26LB = [-0.129289120435714721679687500000, 0.111408457159996032714843750000, -0.499999880790710449218750000000, -0.500000000000000000000000000000, -0.500000000000000000000000000000, ]

def runACAS():
    epsilon = 0.05
    numInputs = 5

    nnet = "ACASXU_run2a_1_1_batch_2000_16bit.pyt"
    nnetPath = nnetPrefix + nnet
    model, is_conv, means, stds = read_net(nnetPath, numInputs, False)
    eran = ERAN(model)

    target = 0

    specUB = prop16UB 
    specLB = prop16LB
    print("Property: " + str(16))
    print("Network: " + nnet)
    res = getEps2(eran, specLB, specUB, target, 5)
    print("Result: " + str(res))
    if abs(res[1]) < epsilon and abs(res[2]) < epsilon:
        print("Verified")
    else:
        print("Failed")

    specUB = prop26UB 
    specLB = prop26LB
    print("Property: " + str(26))
    print("Network: " + nnet)
    res = getEps2(eran, specLB, specUB, target, 5)
    print("Result: " + str(res))
    if abs(res[1]) < epsilon and abs(res[2]) < epsilon:
        print("Verified")
    else:
        print("Failed")



random_pixels = [
[770,148,25,503,664,219,750,351,115,188],
[743,240,781,701,10,458,417,777,130,325],
[408,771,153,438,471,468,87,98,338,624],
[338,696,205,709,303,248,91,449,489,255],
[107,189,524,483,694,219,14,130,759,406],
[359,532,396,538,29,216,422,736,157,4],
[461,182,552,212,647,144,731,629,510,179],
[614,208,520,628,561,339,368,745,611,563],
[434,611,628,481,709,436,582,593,334,457],
[24,556,18,445,459,8,301,86,463,180],
]

def runMNIST3Pixel():
    nnets = [
        nnetPrefix + "/mnist_relu_3_100_16bit.pyt",
        ]
    perturb = 3
    epsilon = 1.0
    numInputs = 784
    numPixels = 3

    csvfile = open('../data/mnist_test.csv', 'r')
    images = csv.reader(csvfile, delimiter=',')
    tests = []
    for i, image in enumerate(images):
        if i == 10: break
        target = int(image[0])
        image = np.float64(image[1:])
        specLB = np.copy(image)
        specUB = np.copy(image)
        for j in range(numPixels):
            specLB[random_pixels[i][j]] = 0.0
            specUB[random_pixels[i][j]] = 255.0
        specLB = specLB / np.float64(255)
        specUB = specUB / np.float64(255)
        tests.append((target, specLB, specUB))

    for nnet in nnets:
        print("Network: " + nnet.split("/")[-1], file=sys.stderr)
        model, is_conv, means, stds = read_net(nnet, numInputs, False)
        eran = ERAN(model)
        for target, lb, ub in tests:
            res = getEps2(eran, lb, ub, target, 10)
            print("Result: " + str(res), file=sys.stderr)
            if abs(res[1]) < epsilon and abs(res[2]) < epsilon:
                print("Verified", file=sys.stderr)
            else:
                print("Failed", file=sys.stderr)
            exit(0)


def runMNIST():
    nnets = [
        nnetPrefix + "/mnist_relu_3_100_16bit.pyt",
        ]
    perturb = 3
    epsilon = 1.0
    numInputs = 784

    csvfile = open('../data/mnist_test.csv', 'r')
    images = csv.reader(csvfile, delimiter=',')
    tests = []
    for i, image in enumerate(images):
        if i == 10: break
        target = int(image[0])
        image = np.float64(image[1:])
        specLB = np.copy(image)
        specUB = np.copy(image)
        specLB = np.clip(specLB - perturb, 0, 255)
        specUB = np.clip(specUB + perturb, 0, 255)
        specLB = specLB / np.float64(255)
        specUB = specUB / np.float64(255)
        tests.append((target, specLB, specUB))

    for nnet in nnets:
        print("Network: " + nnet.split("/")[-1], file=sys.stderr)
        model, is_conv, means, stds = read_net(nnet, numInputs, False)
        eran = ERAN(model)
        for target, lb, ub in tests:
            res = getEps2(eran, lb, ub, target, 10)
            print("Result: " + str(res), file=sys.stderr)
            if abs(res[1]) < epsilon and abs(res[2]) < epsilon:
                print("Verified", file=sys.stderr)
            else:
                print("Failed", file=sys.stderr)
            exit(0)


def runHAR():
    nnet = nnetPrefix + "/HAR_16bit.pyt"

    perturb = 0.1
    epsilon = 0.25
    numInputs = 561

    csvfile = open('../data/har_test.csv', 'r')
    testInputs = csv.reader(csvfile, delimiter=',')
    tests = []
    for i, testInput in enumerate(testInputs):
        if i == 10: break
        target = int(testInput[0])
        testInput = np.float64(testInput[1:])
        specLB = np.copy(testInput)
        specUB = np.copy(testInput)
        specLB = np.clip(specLB - perturb, -1, 1)
        specUB = np.clip(specUB + perturb, -1, 1)
        tests.append((target, specLB, specUB))

    model, is_conv, means, stds = read_net(nnet, numInputs, False)
    eran = ERAN(model)
    for target, lb, ub in tests:
            res = getEps2(eran, lb, ub, target, 6)
            print("Result: " + str(res), file=sys.stderr)
            if abs(res[1]) < epsilon and abs(res[2]) < epsilon:
                print("Verified", file=sys.stderr)
            else:
                print("Failed", file=sys.stderr)
            exit(0)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ["acas", "mnist-global", "mnist-3pixel", "har"]:
        print("usage: python3 " + sys.argv[0] + " EXP")
        print("where EXP is one of 'acas', 'mnist-global', 'mnist-3pixel', or 'har'")
        exit(1)

    exp = sys.argv[1]
    if exp == "acas":
        runACAS()
    elif exp == "mnist-global":
        runMNIST()
    elif exp == "mnist-3pixel":
        runMNIST3Pixel()
    else:
        runHAR()
