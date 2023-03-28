import numpy as np
import jax.numpy as jnp


def increaseSize(input, factor):
    w,h = np.shape(input)
    output = np.zeros((w*factor,h*factor))
    
    for width in range(w*factor):
        for height in range(h*factor):
            output[width][height] = input[width//factor][height//factor]
    return output


def mean_pooling(input,factor):
    """performs a mean pooling operation"""
    
    w,h = np.shape(input)
    if w%factor != 0 or h%factor != 0:
        raise(AssertionError("Non-compatible input shape and downsample factor"))
    
    output = np.zeros((int(w/factor),int(h/factor)))
    
    for width in range(w):
        for height in range(h):
            output[width//factor][height//factor] += input[width][height]
    output /= factor**len(np.shape(output))
    return output

def downsampleHighDefVels(high_def,factor):
    low_def = []
    for vels in high_def:
        both_vels = []
        for vel in vels:
            vel = decreaseSize(vel,factor)

            vel = increaseSize(vel,factor)
            both_vels.append(vel)
        low_def.append(both_vels)
    return low_def

def sampling(input,factor):
    """performs sampling operation"""
    
    w,h = np.shape(input)
    if w%factor != 0 or h%factor != 0:
        raise(AssertionError("Non-compatible input shape and downsample factor"))
    
    output = np.zeros((int(w/factor),int(h/factor)))
    #print(np.shape(output))
    
    for width in range(0,w,factor):
        for height in range(0,h,factor):
            output[width//factor][height//factor] = input[width][height]
            
    return output
    

def creatingDataset(highDef,method,factor):
    """
    Performs ´method´ on contents of high_def
    
    e.g. method = mean_pooling
    This function performs a mean pooling operation on each frame within high_def
    
    """
    
    low_def = []
    for vels in highDef:
        
        u = method(np.array(vels[:,:,0]),factor) #output is a np.array
        
        
        v = method(np.array(vels[:,:,1]),factor) #output is a np.array
        
        
        low_def.append(jnp.dstack([
            u,
            v
        ]))
    return low_def


def calculateResiduals(X,Y):
    out = []
    for i in range(len(X)):
        out.append(Y[i]-X[i])
    return out


def normalisingDataset(data):
    theMean = np.mean(data)
    theStdDev = np.std(data)
    for i in range(len(data)):
        data[i] = (data[i]-theMean)/theStdDev
    return data,theMean,theStdDev

def deNormalising(data,theMean,theStdDev):
    for i in range(len(data)):
        data[i] = data[i]*theStdDev+theMean
    return data