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

#This is way too slow, learn to use jax.jit and maybe make it faster, for now use the numpy implementation below
def calculateALLDerivativesJAX(data,size,domain,postprocess,factor):
    mygrid = cfd.grids.Grid(size,domain=domain)
    
    data = reshapeData(test,mygrid)

    return jnp.dstack([
            postprocess(data[0].data,factor),#u
            postprocess(data[1].data,factor),#v

            postprocess(fd.laplacian(data[0]).data,factor), #lap(u)
            postprocess(fd.forward_difference(data[0],axis=0).data,factor), #dudx
            postprocess(fd.forward_difference(data[0],axis=1).data,factor), #dudy

            postprocess(fd.laplacian(data[1]).data,factor), #lap(v)
            postprocess(fd.forward_difference(data[1],axis=0).data,factor), #dvdx
            postprocess(fd.forward_difference(data[1],axis=1).data,factor) #dvdy
        ])

def npLaplacian(data):
    """
    ddx^2 + ddy^2
    """
    
    ddx2 = np.gradient(np.gradient(data,axis=1),axis=1)
    ddy2 = np.gradient(np.gradient(data,axis=0),axis=0)
    
    return ddx2 + ddy2

def calculateALLDerivativesNUMPY(data,postprocess,factor):
    """
    Calculates derivatives and laplacians of velocity fields and returns them stacked in a DeviceArray in the following format:
    
        u
        dudx
        dudy
        lap(u)

        v
        dvdx
        dvdy
        lap(v)
    
    Uses numpy methodologies, returns jax.numpy datatype
    """
    
    
    #u
    u = data[:,:,0]
    du = np.gradient(u)
    lapu = npLaplacian(u)
    
    #v
    v = data[:,:,1]
    dv = np.gradient(v)
    lapv = npLaplacian(v)
    
    #postprocess
    if postprocess is not None:
        u = postprocess(u,factor)
        du[0] = postprocess(du[0],factor)
        du[1] = postprocess(du[1],factor)
        lapu = postprocess(lapu,factor)
        
        v = postprocess(v,factor)
        dv[0] = postprocess(dv[0],factor)
        dv[1] = postprocess(dv[1],factor)
        lapv = postprocess(lapv,factor)
        
    
    return jnp.dstack([
            u,
            du[1],
            du[0],
            lapu,

            v,
            dv[1],
            dv[0],
            lapv
        ])


def createDatasetDerivatives(dataset,postprocess,factor):
    out = []
    for i in range(len(dataset)):
        out.append(calculateALLDerivativesNUMPY(dataset[i],postprocess,factor))
    return out