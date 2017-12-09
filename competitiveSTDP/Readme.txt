#########################################################################
Author: Zhengzhong Liang, David Schwartz
Email Address: {zhengzhongliang, dmschwar}@email.arizona.edu

Readme: competitive STDP 
#########################################################################

This directory contains all source code required to run the simulations from which certain results in the machine learning final project paper are derived. 

MainFunction.py controls global parameters and the course of the simulation
Classes.py contains object oriented implementations of the simulation environement, neurons, neural networks, their dynamics etc. Neuron and neural network parameters are defined here.
Other source files, such as MNIST.py, ReadDataFunction.py, etc. implement supplementary functionality used to plot and record observations made on simulation results.


Note:
1, This program is used to conduct our competitive STDP learning experiment. 
2, A subroutine found herein implements classical STDP. However, this is not used to produce the results relating to classical STDP in the paper.