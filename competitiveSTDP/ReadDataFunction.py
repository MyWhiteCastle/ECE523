from skimage import filter as filt
import pickle
import numpy as np

def readData():
#---------------------------------------------
    with open('train0.pkl', 'rb') as input:
        train0 = pickle.load(input)
    with open('test0.pkl', 'rb') as input:
        test0 = pickle.load(input)

#---------------------------------------------
    with open('train1.pkl', 'rb') as input:
        train1 = pickle.load(input)
    with open('test1.pkl', 'rb') as input:
        test1 = pickle.load(input)

#---------------------------------------------
    with open('train2.pkl', 'rb') as input:
        train2 = pickle.load(input)
    with open('test2.pkl', 'rb') as input:
        test2 = pickle.load(input)

#---------------------------------------------
    with open('train3.pkl', 'rb') as input:
        train3 = pickle.load(input)
    with open('test3.pkl', 'rb') as input:
        test3 = pickle.load(input)

#---------------------------------------------
    with open('train4.pkl', 'rb') as input:
        train4 = pickle.load(input)
    with open('test4.pkl', 'rb') as input:
        test4 = pickle.load(input)

#---------------------------------------------
    with open('train5.pkl', 'rb') as input:
        train5 = pickle.load(input)
    with open('test5.pkl', 'rb') as input:
        test5 = pickle.load(input)

#---------------------------------------------
    with open('train6.pkl', 'rb') as input:
        train6 = pickle.load(input)
    with open('test6.pkl', 'rb') as input:
        test6 = pickle.load(input)

#---------------------------------------------
    with open('train7.pkl', 'rb') as input:
        train7 = pickle.load(input)
    with open('test7.pkl', 'rb') as input:
        test7 = pickle.load(input)

#---------------------------------------------
    with open('train8.pkl', 'rb') as input:
        train8 = pickle.load(input)
    with open('test8.pkl', 'rb') as input:
        test8 = pickle.load(input)

#---------------------------------------------
    with open('train9.pkl', 'rb') as input:
        train9 = pickle.load(input)
    with open('test9.pkl', 'rb') as input:
        test9 = pickle.load(input)

    return train0,train1,train2,train3,train4,train5,train6,train7,train8,train9,test0,test1,test2,test3,test4,test5,test6,test7,test8,test9

def readCollectedSpikes():
    with open('trainSpikesTensor2.pkl', 'rb') as input:
        trainSpikesTensor = pickle.load(input)
    with open('testSpikesTensor2.pkl', 'rb') as input:
        testSpikesTensor = pickle.load(input)
    return trainSpikesTensor,testSpikesTensor
