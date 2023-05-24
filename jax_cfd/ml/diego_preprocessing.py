import numpy as np
import jax.numpy as jnp

from jax_cfd.ml.diego_cnn_bcs import retrieveField, createPaddedMesh, channelFlowPadding


    
def increaseSize(input,factor):
    w,h = np.shape(input)
    output = np.zeros((w*factor,h*factor))

    for width in range(w*factor):
        for height in range(h*factor):
            output[width][height] = input[width//factor][height//factor]
    return output


def decreaseSize(input,factor):
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
    print("WARNING: Use downsampleHighDefVelsNumpy for better performance")
    low_def = []
    for vels in high_def:
        both_vels = []
        for vel in vels:
            vel = decreaseSize(vel,factor)

            vel = increaseSize(vel,factor)
            both_vels.append(vel)
        low_def.append(both_vels)
    return low_def

def downsampleHighDefVelsNumpy(high_def,factor):
    """
    Same as downsampleHighDefVels but running the resizings on numpy arrays
    due to performance issues with these functions and DeviceArrays (jnp arrays)
    """
    low_def = []
    for vels in high_def:

        u = decreaseSize(np.array(vels[:,:,0]),factor) #output is a np.array
        u = jnp.array(increaseSize(u,factor)) #output is a DeviceArray

        v = decreaseSize(np.array(vels[:,:,1]),factor) #output is a np.array
        v = jnp.array(increaseSize(v,factor)) #output is a DeviceArray

        low_def.append(jnp.dstack([
            u,
            v
        ]))
    return low_def


# def increaseSize(input, factor):
#     w,h = np.shape(input)
#     output = np.zeros((w*factor,h*factor))
    
#     for width in range(w*factor):
#         for height in range(h*factor):
#             output[width][height] = input[width//factor][height//factor]
#     return output


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

# def downsampleHighDefVels(high_def,factor):
#     low_def = []
#     for vels in high_def:
#         both_vels = []
#         for vel in vels:
#             vel = decreaseSize(vel,factor)

#             vel = increaseSize(vel,factor)
#             both_vels.append(vel)
#         low_def.append(both_vels)
#     return low_def

# def sampling(input,factor):
#     """performs sampling operation"""
    
#     w,h = np.shape(input)
#     if w%factor != 0 or h%factor != 0:
#         raise(AssertionError("Non-compatible input shape and downsample factor"))
    
#     output = np.zeros((int(w/factor),int(h/factor)))
#     #print(np.shape(output))
    
#     for width in range(0,w,factor):
#         for height in range(0,h,factor):
#             output[width//factor][height//factor] = input[width][height]
            
#     return output


def sampling(data,factor):
    return data[0::factor,0::factor]


# def newSampling(input,factor,precomputed):
#     """performs sampling operation"""
    
    
    
#     output = np.zeros((precomputed["wf"],precomputed["hf"]))
#     #print(np.shape(output))
    
#     for width in range(0,precomputed["w"],factor):
#         for height in range(0,precomputed["h"],factor):
#             output[precomputed["wff"]-1][precomputed["hff"]-1] = input[precomputed["w"]][precomputed["h"]]
            
#     return output
    

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

def calculateALLDerivativesNUMPY(data,postprocess,factor,which_outputs):
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
    
    postprocess accepts a jnp.array and a factor (a sampling step)
    
    which_outputs = {
        "u" : True,
        "du" : True,
        "lapu" : True,
        
        "v" : True,
        "dv" : True,
        "lapv" : True
    }
    """
    if postprocess is None:
        postprocess = lambda x,factor : x
    
    output_list = []
    #u
    u = data[:,:,0]
    if which_outputs["u"] is True:
        output_list.append(postprocess(u,factor))
    
    if which_outputs["du"] is True:
        du = np.gradient(u)
        du[0] = postprocess(du[0],factor)
        du[1] = postprocess(du[1],factor)
        output_list.append(du[0])
        output_list.append(du[1])
    
    if which_outputs["lapu"] is True:
        lapu = npLaplacian(u)
        lapu = postprocess(lapu,factor)
        output_list.append(lapu)
    
    #v
    v = data[:,:,1]
    if which_outputs["v"] is True:
        output_list.append(postprocess(v,factor))
        
    if which_outputs["dv"] is True:
        dv = np.gradient(v)
        dv[0] = postprocess(dv[0],factor)
        dv[1] = postprocess(dv[1],factor)
        output_list.append(dv[0])
        output_list.append(dv[1])
        
    if which_outputs["lapv"] is True:
        lapv = npLaplacian(v)
        lapv = postprocess(lapv,factor)
        output_list.append(lapv)
    

    
    return jnp.dstack(output_list)


def createDatasetDerivatives(dataset,postprocess,factor,which_outputs):
    out = []
    for i in range(len(dataset)):
        out.append(calculateALLDerivativesNUMPY(dataset[i],postprocess,factor,which_outputs))
    return out



# new way to apply bcs, apply to u and v and infer for the rest
def retrieveAll(data,padding):
    
    myshape = np.shape(data)
    out = []
    for time in range(myshape[0]):
        partial = []
        for channel in range(myshape[-1]):
            partial.append(retrieveField(data[time][:,:,channel],padding[0],padding[1]))
        out.append(jnp.dstack(partial))

    return out

def sampleAll(data,factor,postprocess = sampling):
    myshape = np.shape(data)
    out = []
    for time in range(myshape[0]):
        partial = []
        for channel in range(myshape[-1]):
            partial.append(postprocess(data[time][:,:,channel],factor))
        out.append(jnp.dstack(partial))
    return out
def yDatasetOneTime(dataIN,which_outputs,padding,factor):
    
    # here we apply the bcs
    u = dataIN[:,:,0]
    
    u = createPaddedMesh(u,padding)
    
    v = dataIN[:,:,1]
    v = createPaddedMesh(v,padding)
#     print(u)
    
    u = channelFlowPadding(u,padding,0,0)
    v = channelFlowPadding(v,padding,0,0)
#     print(u)
            
    
    output_list = []
    if which_outputs["vels"]:
        output_list.append(u)
        output_list.append(v)
    
    if which_outputs["derivatives"]:
        du = np.gradient(u)
        du = jnp.array(du)
        output_list.append(du[0])
        output_list.append(du[1])
        
        dv = np.gradient(v)
        dv = jnp.array(dv)
        output_list.append(dv[0])
        output_list.append(dv[1])
    
    if which_outputs["laplacians"]:
        lapu = npLaplacian(u)
        lapu = jnp.array(lapu)
        output_list.append(lapu)
        
        lapv = npLaplacian(v)
        lapv = jnp.array(lapv)
        output_list.append(lapv)
    
    return jnp.dstack(output_list)
    
    
def createYDatasetNew(data,which_outputs,padding,factor):
    w,h,_ = np.shape(data[0])
    precomputed = {
        "w" : w,
        "h" : h,
        "wf" : int(w/factor),
        "hf" : int(h/factor),
        "wff" : w//factor,
        "hff" : h//factor
    }
    
    out = []
    j = 0
    for i in data:
        out.append(yDatasetOneTime(i,which_outputs,padding,factor))
        j += 1
        
    return out

def getYdata(data,which_outputs,padding,postprocess,factor):
    Y = createYDatasetNew(data,which_outputs,padding,factor) #pads and calculates derivatives
    
    Y = retrieveAll(Y,padding) #removes padding
    
    Y = sampleAll(Y,factor,postprocess) #samples data to coarse grid dimensions
    
    return Y
