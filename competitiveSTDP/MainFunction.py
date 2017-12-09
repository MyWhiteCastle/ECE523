import numpy as np
from ReadDataFunction import *
from classes import *
from ConvertToSpikes import *
import matplotlib.pyplot as plt
from ToolFunction import *
from sklearn.model_selection import KFold


#Author: zhengzhong liang and David S.
#Course: ECE523 
#Date: 2017-05-03


#1 This is the main code to fulfill the experiment. It needs python 2.7 with Anaconda to compile
#this version is used for generating cross validation result, which is in paper
#note that the version can be modified w.r.t. learning window and number of folds
# even 1 fold validation may take as long as 10 minutes to run.
#If this code does not compile, please feel free to contact me.

trainSpikesTensor,testSpikesTensor=readCollectedSpikes()
datasetVolume=100
folds=10
#---------------generate data for kflod cross validation-----------------
X=np.zeros([datasetVolume*2,784])
y=np.zeros(datasetVolume*2)
for i in np.arange(datasetVolume):
    X[i*2,:]=trainSpikesTensor[0,i,:]
    y[i*2]=0
    X[i*2+1,:]=trainSpikesTensor[1,i,:]
    y[i*2+1]=1
kf=KFold(n_splits=folds)

#----------------create network component--------------------------------
inputGroup=InputGroup(N=784,indices=[],times=[])   #create input signal
neuronGroup=NeuronGroup(N=2)                      #create output neuron
synapses=Synapses(inputGroup,neuronGroup)          #create
teachSignal=TeachSignal(neuronGroup=neuronGroup,indices=[],times=[])
network=Network(inputGroupList=[inputGroup],neuronGroupList=[neuronGroup],synapsesList=[synapses],teachSignalList=[teachSignal])
simulation=Simulation(network)
#-----------------training phase----------------------
#trainingSampleNumber=25  #use 25 image in each class to train
trainingDuration=800
timeStep=1

for i in np.arange(len(synapses.taus)):
    synapses.taus[i]=(784-i)/4+100

sumAccuracy=0
for train, test in kf.split(X):      #do 10 folds cross-validation
    X_train=X[test]
    y_train=y[test]
    X_test=X[train]
    y_test=y[train]
    for i in np.arange(len(y_train)):     # training start
        digit=int(y_train[i])
        print('digit: ',digit)
        spikes=np.zeros(2)
        while spikes[digit]<3: #or (spikes[digit]<=spikes[1-digit]):
            inputIndices,inputTimes=generateInputSignal(X_train[i])
            targetIndices,targetTimes=generateTargetSignal(digit)    #train untill target neuron spieks more than 3 times
            inputGroup.indices=inputIndices
            inputGroup.times=inputTimes
            teachSignal.indices=targetIndices
            teachSignal.times=targetTimes
            synapses.learning=True
            spikes=simulation.train(trainingDuration,timeStep,digit)
            #plt.figure()
            #plt.subplot(2,1,1)
            #plt.plot(w126)
            #plt.subplot(2,1,2)
            #plt.plot(w712)
            network.resetNetwork()
            #print('W712:'spikes)
            #print('spikes:',spikes)

    oneFoldCount=0
    for j in np.arange(len(y_test)):       #test
        digit=int(y_test[j])
        spikes=np.zeros(2)
        inputIndices,inputTimes=generateInputSignal(X_test[j])
        targetIndices,targetTimes=generateTargetSignal(digit)
        inputGroup.indices=inputIndices
        inputGroup.times=inputTimes
        teachSignal.indices=[]
        teachSignal.times=[]
        synapses.learning=False
        spikes=simulation.test(trainingDuration,timeStep)
        if spikes[digit]>spikes[1-digit]:
            print('Hit!')
            oneFoldCount=oneFoldCount+1
        print('spikes,',spikes,'   digit,',digit)
        network.resetNetwork()
    oneFoldAccuracy=oneFoldCount/1.0/len(y_test)
    print('1 Fold Accuracy:',oneFoldAccuracy)
    synapses.w=0.05*synapses.wMax*np.ones([len(neuronGroup.s),len(inputGroup.s)])
    sumAccuracy=(sumAccuracy+oneFoldAccuracy)/1.0

tenFoldAccuracy=sumAccuracy/1.0/folds

print('10-Fold Accuracy is: ',tenFoldAccuracy)


#plt.show()
        #plt.figure()
        #plt.subplot(3,1,1)
        #plt.plot(v1)
        #plt.subplot(3,1,2)
        #plt.plot(v2)
        #plt.subplot(3,1,3)
        #plt.plot(v3)


#-----------------testing phase-----------------------
#testingSampleNumber=5   #use 10 images in each class to test
#testingDruation=80
#timeStep=0.1
#print('Testing Phase Start\n-------------------\n\n')
#for digit in np.arange(2):
#    print('Testing Digit ',digit)
#    for sample in np.arange(testingSampleNumber):
#        print('Testing Samples,',sample)
#        inputIndices,inputTimes=generateInputSignal(testSpikesTensor[digit,sample,:])
#        targetIndices,targetTimes=generateTargetSignal(digit)
#        inputGroup.indices=inputIndices
#        inputGroup.times=inputTimes
#        teachSignal.indices=[]
#        teachSignal.times=[]
#        synapses.learning=False
#        spikes=simulation.test(trainingDuration,timeStep)
        #plt.figure()
        #plt.subplot(3,1,1)
        #plt.plot(v1)
        #plt.subplot(3,1,2)
        #plt.plot(v2)
        #plt.subplot(3,1,3)
        #plt.plot(v3)
#        print('Output Spikes:',spikes)
#        network.resetNetwork()

#plt.show()


