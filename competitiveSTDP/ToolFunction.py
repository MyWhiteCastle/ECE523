import numpy as np

def compare(array,digit):
    digitSpike=array[digit]
    newArray=np.delete(array,digit)
    secondMax=np.amax(newArray)
    if digitSpike>secondMax:
        return True
    else:
        return False




