import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
#from arch import arch_model
from seaborn import tsplot

import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.random import rand

def rasterPlot(s,timeStep):
    t=[]
    index=[]
    for i,j in enumerate(s):
        if j==1:
            t.append(i*timeStep)
            index.append(0)
    plt.scatter(t,index)

