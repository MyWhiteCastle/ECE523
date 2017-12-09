import math as m
import numpy as np
import random
from PIL import Image
from sklearn import decomposition
from skimage import filter as filt
import numpy as np
from numpy.random import rand
#-------------neural network class-------------------------------------------------------
class NeuronGroup:  #what does this object means?
    def __init__(self,N=1,Vrest=-70.,Vthre=-55.,Vreset=-74.,taum=150.):   #when dt=0.1, taum =15
        self.Vrest=Vrest
        self.Vthre=Vthre
        self.Vreset=Vreset
        self.taum=taum
        self.v=Vrest*np.ones(N)
        #self.E=np.zeros(N)
        self.s=np.zeros(N)
        self.parent=None
        self.child=None
class Synapses:
    def __init__(self,pre_group,post_group,efficacy=10,learning=True):
        N_pre=len(pre_group.s)
        N_post=len(post_group.s)
        self.wMax=3
        self.wCons=5*self.wMax
        self.efficacy=efficacy  #efficacy of synapse. i.e. how much will the
        self.w=0.05*self.wMax*np.ones([N_post,N_pre])#+0.1*rand(N_post,N_pre)
        self.Iext=np.zeros([N_post,N_pre])
        self.taun=40*np.ones([N_post,N_pre])#+rand(N_post,N_pre)  #when dt=0.1, taun=4
        self.sTime_pre=np.zeros(N_pre)*1.0
        self.sTime_post=np.zeros(N_post)*1.0
        self.parent=pre_group
        self.child=post_group
        self.parent.child=self
        self.child.parent=self
        self.learning=learning
        self.Apre=0.02*self.wMax    #maximum learning speed
        self.Apost=0.1*self.wMax    #maximum learning speed
        self.taus=200*np.ones(N_pre)   #decaying coefficient of STDP  #when dt=0.1, taus= 20
    def updatePre_classicalSTDP(self,index):
        post_neurons=self.child
        N_post=len(post_neurons.s)
        for i in np.arange(N_post):
            timeDifference=self.sTime_pre[index]-self.sTime_post[i]
            if (np.abs(self.sTime_post[i]-0)>0.05):
                delta_w=self.Apre*np.exp(-timeDifference/self.taus[index])
                self.w[i,index]=np.clip(self.w[i,index]-delta_w,0,self.wMax)
    def updatePost_classicalSTDP(self,index):
        pre_neurons=self.parent
        N_pre=len(pre_neurons.s)
        for i in np.arange(N_pre):
            timeDifference=self.sTime_post[index]-self.sTime_pre[i]
            if (np.abs(self.sTime_pre[i]-0)>0.05):
                delta_w=self.Apost*np.exp(-timeDifference/self.taus[i])
                self.w[index,i]=np.clip(self.w[index,i]+delta_w,0,self.wMax)
    def updatePost_competitiveSTDP(self,index):
        pre_neurons=self.parent
        post_neurons=self.child
        N_pre=len(pre_neurons.s)
        N_post=len(post_neurons.s)
        for i in np.arange(N_pre):
            timeDifference=self.sTime_post[index]-self.sTime_pre[i]
            if (np.abs(self.sTime_pre[i]-0)>0.05):
                delta_w=self.Apost*np.exp(-timeDifference/self.taus[i])
                self.w[index,i]=np.clip(self.w[index,i]+delta_w,0,self.wMax)
                for j in np.arange(N_post):
                    if(j!=index):
                        #self.w[j,i]=np.clip(self.w[j,i]*0.999,0,self.wMax)
                        self.w[j,i]=np.clip(self.w[j,i]*(1-delta_w/(self.wCons-self.w[index,i])),0,self.wMax)

class InputGroup:
    def __init__(self,N,indices,times):
        self.s=np.zeros(N)
        self.indices=indices
        self.times=times  #must be 2-D np array
        self.child=None

class TeachSignal:
    def __init__(self,neuronGroup,indices,times):
        self.objectNeuron=neuronGroup
        self.indices=indices
        self.times=times
        if self.objectNeuron.parent==None:
            self.objectNeuron.parent=self

class Network:
    def __init__(self,inputGroupList=[],neuronGroupList=[],synapsesList=[],teachSignalList=[]):   #must be python list
        self.inputGroupList=inputGroupList
        self.neuronGroupList=neuronGroupList
        self.synapsesList=synapsesList
        self.teachSignalList=teachSignalList
    def resetNetwork(self):
        for neuronGroup in self.neuronGroupList:
            size=len(neuronGroup.s)
            neuronGroup.v=neuronGroup.Vrest*np.ones(size)
            neuronGroup.s=np.zeros(size)
        for synapses in self.synapsesList:
            N_pre=len(synapses.parent.s)
            N_post=len(synapses.child.s)
            synapses.Iext=np.zeros([N_post,N_pre])
            synapses.sTime_pre=np.zeros(N_pre)
            synapses.sTime_post=np.zeros(N_post)
        for inputGroup in self.inputGroupList:
            inputGroup.s=np.zeros(len(inputGroup.s))
    def updateNetwork(self,timeStep,currentTime,digit):
        #-------update neuron state----------------------
        for inputGroup in self.inputGroupList:
            N=len(inputGroup.indices)
            for i in np.arange(N):
                if (np.abs(inputGroup.times[i]-currentTime)<0.01):
                    inputGroup.s[inputGroup.indices[i]]=1
                    #-----update the spike times array in Synapse object-------
                    #inputGroup.child.sTime_pre[i]=currentTime
        #--------update teach signal------------------------------------
        for teachSignal in self.teachSignalList:
            objectNeuron=teachSignal.objectNeuron
            N=len(teachSignal.indices)
            for i in np.arange(N):
                if (np.abs(teachSignal.times[i]-currentTime)<0.01):
                    objectNeuron.v[teachSignal.indices[i]]=objectNeuron.Vthre+0.5
        #--update Iext matrix. This matrix reflext how weight affect post-synaptic activities!------------
        for neuronGroup in self.neuronGroupList:
            if (neuronGroup.parent==None):
                neuronGroup.v=neuronGroup.v+(neuronGroup.Vrest-neuronGroup.v)/neuronGroup.taum*timeStep
            else:
                pre_neurons=neuronGroup.parent.parent
                synapse=neuronGroup.parent
                post_neurons=neuronGroup
                N_pre=len(pre_neurons.s)
                N_post=len(post_neurons.s)
                #-----step 1: update decaying Iext-------
                for i in np.arange(N_post):
                    for j in np.arange(N_pre):
                        synapse.Iext[i,j]=synapse.Iext[i,j]-synapse.Iext[i,j]/synapse.taun[i,j]*timeStep
                        #----step 2: check pre-synaptic spikes--
                        if pre_neurons.s[j]==1:
                            #update sTime_pre here: !!!
                            synapse.Iext[i,j]=synapse.Iext[i,j]+synapse.w[i,j]*synapse.efficacy
        #-----------update neuron activities and synapse weights------------------------------------------
        for neuronGroup in self.neuronGroupList:
            if (neuronGroup.child==None):
                for i in np.arange(len(neuronGroup.s)):
                    if neuronGroup.s[i]==1:
                        neuronGroup.s[i]=0
                        neuronGroup.v[i]=neuronGroup.Vreset
            if (neuronGroup.parent!=None):
                pre_neurons=neuronGroup.parent.parent
                synapse=neuronGroup.parent
                post_neurons=neuronGroup
                N_pre=len(pre_neurons.s)
                N_post=len(post_neurons.s)
                #----step 3: clear pre-synaptic spike flag-----
                for j in np.arange(N_pre):
                    if pre_neurons.s[j]==1:
                        #step 1: update sTime_pre array in synapse
                        synapse.sTime_pre[j]=currentTime-timeStep
                        #step 2: update the synapse weight
                        if synapse.learning==True:
                            synapse.updatePre_classicalSTDP(j)
                        #step 3: clear the spike flag
                        pre_neurons.s[j]=0
                #----step 4: update post-neuron ---------------
                for i in np.arange(N_post):
                    Iext_sum=np.sum(synapse.Iext[i])
                    post_neurons.v[i]=post_neurons.v[i]+(post_neurons.Vrest+Iext_sum-post_neurons.v[i])/post_neurons.taum*timeStep
                    if post_neurons.v[i]>=post_neurons.Vthre:
                        post_neurons.v[i]=post_neurons.Vreset
                        post_neurons.s[i]=1
                        #step 1: update sTime_post array in synapse
                        synapse.sTime_post[i]=currentTime
                        #step 2: update the synapse weight
                        if ((synapse.learning==True) and (i==digit)):
                            synapse.updatePost_competitiveSTDP(i)
                            #synapse.updatePost_classicalSTDP(i)
        #add code here about what data you want to track
        #if self.neuronGroupList[0].s[digit]==1:
        #    spike_time=currentTime
        #else:
        #    spike_time=[]
        return self.neuronGroupList[0].s

class Simulation:
    def __init__(self,network):
        self.network=network
        #-----test variable------
        self.result=[]
        #------------------------
    def train(self,tMax,timeStep=0.1,digit=0):
        #w126=[]
        #w712=[]
        #spikeTime=[]
        #spikesCount=np.zeros(2)
        spikesCount=np.zeros(2)
        for i in np.arange(tMax/timeStep):
            t=i*timeStep
            spikes=self.network.updateNetwork(timeStep,t,digit)
            spikesCount=spikesCount+spikes
            #v1.append(voltage[0])
            #v2.append(voltage[1])
            #v3.append(voltage[2])
            #w126.append(w[0][126])
            #w712.append(w[0][712])
            #spikeTime.append(spike_time)
        return spikesCount
    def test(self,tMax,timeStep=0.1,digit=0):
        v1=[]
        v2=[]
        v3=[]
        spikesCount=np.zeros(2)
        for i in np.arange(tMax/timeStep):
            t=i*timeStep
            spikes=self.network.updateNetwork(timeStep,t,digit)
            spikesCount=spikesCount+spikes
            #v1.append(voltage[0])
            #v2.append(voltage[1])
            #v3.append(voltage[2])
        return spikesCount#,v1,v2,v3

class Result:
    def __init__(self):
        self.monitor=[]

#------------image class-----------------------------------------------------------------
class MNISTImage(object):

    def __init__(self, digit, image, data=None):
        self.digit = digit
        self.image = image
        self.data = data
        self.encoded_data = self.encode()

    def encode(self):
        data_array = np.array(self.image)#data).reshape((28,28))
        edges = filt.canny(data_array, sigma=3)
        encoded = np.array(edges).reshape(784)
        return encoded
