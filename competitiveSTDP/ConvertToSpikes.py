import numpy as np
from ReadDataFunction import *
import pickle
from skimage import filter as filt

def generateInputSignal(sample):
    indices=[]
    times=[]
    for i in np.arange(len(sample)):
        if sample[i]==1:
            indices.append(i)
            times.append((i+1))
    return indices,times
def generateTargetSignal(digit):
    return np.array([digit]), np.array([785])

def collectSpikes(trainingSampleNumber,testingSampleNumber):   #should be less than 1000
    images=readData()
    # read training data:
    trainSpikesTensor=np.ones([10,trainingSampleNumber,784])
    for digits in np.arange(10):
        trainingSheet=np.empty([trainingSampleNumber,784])
        dataSet=images[digits]
        for samples in np.arange(trainingSampleNumber):
            trainingSheet[samples,:]=dataSet[samples].encoded_data
        trainSpikesTensor[digits,:,:]=trainingSheet

    # read testing data:
    testSpikesTensor=np.ones([10,testingSampleNumber,784])
    for digits in np.arange(10):
        testingSheet=np.empty([testingSampleNumber,784])
        dataSet=images[digits+10]
        for samples in np.arange(testingSampleNumber):
            testingSheet[samples,:]=dataSet[samples].encoded_data
        testSpikesTensor[digits,:,:]=testingSheet
    return trainSpikesTensor,testSpikesTensor

def writeSpikeTensor(trainSpikesTensor,testSpikesTensor):
    with open('trainSpikesTensor2.pkl', 'wb') as output:
        pickle.dump(trainSpikesTensor, output, pickle.HIGHEST_PROTOCOL)
    with open('testSpikesTensor2.pkl', 'wb') as output:
        pickle.dump(testSpikesTensor, output, pickle.HIGHEST_PROTOCOL)

#trainSpikesTensor,testSpikesTensor=collectSpikes(100,100)
#writeSpikeTensor(trainSpikesTensor,testSpikesTensor)



#  Tensor[X,Y,Z]: X: number of classes = 0~9
#                 Y: number of images in each class: about 1000 samples in each class
#                 Z: 784 pixels in each image



