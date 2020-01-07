import sys
import numpy as np
import readNNet

def writePyt(weight, biases, fileName):
    f = open(fileName, 'w')
    print(len(weight))
    for i in range(len(weight) - 1):
        #if i == len(weight) - 2:
        #    f.write('Affine\n')
        #else:
        f.write('ReLU\n')
        print(np.array(weight[i]).shape)
        weightLayer = np.array(weight[i])
        weightLayer = np.transpose(weightLayer)
        weightLayer.astype(np.float128)
        f.write('[')
        for j in range(weightLayer.shape[0]):
            if j > 0:
                f.write(', [')
            else:
                f.write('[')
            for k in range(weightLayer.shape[1]):
                if k > 0:
                    f.write(', ')
                f.write(str(weightLayer[j][k]))
            f.write(']')
        f.write(']\n')                
        print(np.array(biases[i]).shape)
        biasesLayer = biases[i]
        biasesLayer.astype(np.float128)
        f.write('[')
        for j in range(biases[i].shape[0]):
            if j > 0:
                f.write(', ')
            f.write(str(biasesLayer[j]))
        f.write(']\n')
    f.close()

def getFileName(fileName):
    name = fileName.split('.')
    print(name[0])
    fileNameOut = name[0] + '.pyt'
    print(fileNameOut)
    return fileNameOut

if __name__=='__main__':
    weights, biases = readNNet.readNNet(sys.argv[1])
    writePyt(weights, biases, getFileName(sys.argv[1]))

