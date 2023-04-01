import jax.numpy as jnp
import jax


## Utilities
def findPadding(kernel):
    
    padding = []
    for kernel_length in jnp.shape(kernel):
        if kernel_length % 2 == 0:
            raise AssertionError('Kernel must have odd lengths in each dimension')
        padding.append(kernel_length //2)
    
    return padding

def createPaddedMesh(mesh,padding):

    rowPad = padding[0]
    colPad = padding[1]
        
    (rows,cols) = jnp.shape(mesh)

    paddedMesh = jnp.zeros((rows + 2*padding[0],
                           cols + 2*padding[1]))
    return paddedMesh.at[rowPad:-rowPad,colPad:-colPad].set(mesh)

def createPaddedMesh_jit(mesh,kernel):
    
    padding = findPadding(kernel)
    rowPad = padding[0]
    colPad = padding[1]
        
    (rows,cols) = jnp.shape(mesh)

    paddedMesh = jnp.zeros((rows + 2*padding[0],
                           cols + 2*padding[1]))
    return paddedMesh.at[rowPad:-rowPad,colPad:-colPad].set(mesh)

createPaddedMesh_jit = jax.jit(createPaddedMesh_jit)



## Paddings for each boundary condition
def periodicPadding(data,pad,axis=0):
    """
    implements periodic padding to both ends of given dimension
    
    axis=0 -> left and right
    axis=1 -> top and bottom (transpose then left and right then transpose)
    """
    if axis == 1:
        data = data.T

    data = data.at[:,:pad].set(data.at[:,-2*pad:-pad].get())

    data = data.at[:,-pad:].set(data.at[:,pad:2*pad].get())

    if axis == 1:
        data = data.T
        
    return data

def dirichletPadding(data,pad,leftPad,rightPad,axis=0):
    """
    implements dirichlet padding to both ends of given dimension
    
    axis=0 -> left and right
    axis=1 -> top and bottom (transpose, then left and right, then transpose back)
    """
    if axis == 1:
        data = data.T

    data = data.at[:,:pad].set(leftPad)

    data = data.at[:,-pad:].set(rightPad)

    if axis == 1:
        data = data.T
        
    return data

def padCorners(data,pad,value):
    
    
    data = data.at[0,0].set(value)
    data = data.at[0,-1].set(value)
    
    data = data.at[-1,0].set(value)
    data = data.at[-1,-1].set(value)
    
    return data

def channelFlowPadding(data,padding,topWall,lowWall):
#     (padRow,padCol) = findPadding(kernel)
    padRow = padding[0]
    padCol = padding[1]
    
    if padRow != padCol:
        raise AssertionError('Only square filters supported')
    
    data = periodicPadding(data,padRow,axis = 0)
    
    
    data = dirichletPadding(data,padCol,topWall,lowWall,axis = 1)
    
    return padCorners(data,padRow,0)

def retrieveField(data,kernel):
    (padRow,padCol) = findPadding(kernel)
    return data.at[padRow:-padRow,padCol:-padCol].get()





#these pad the datasets after the derivatives have been calculated
def padXDataset(dataset,padding):
    """
    Only u and v velocities
    """
    times = len(dataset)
    out = []
    temp = createPaddedMesh(dataset[0][:,:,0],padding)
    for i in range(times):
        temp1 = createPaddedMesh(dataset[i][:,:,0],padding)
        temp2 = createPaddedMesh(dataset[i][:,:,1],padding)
        out.append(jnp.dstack([
                channelFlowPadding(temp1,padding,0,0),
                channelFlowPadding(temp2,padding,0,0)
            ]))
            
            
    return out

def padYDataset(dataset,padding,conditions=None):
    """
    u
    dudx
    dudy
    lap(u)

    v
    dvdx
    dvdy
    lap(v)
    """
    if conditions is None:
        conditions = {
            "u" : [0,0],
            "dudx" : [0,0],
            "dudy" : [0,0],
            "lap(u)" : [0,0],

            "v" : [0,0],
            "dvdx" : [0,0],
            "dvdy" : [0,0],
            "lap(v)" : [0,0]
        }
    
    #read them once
    [uT,uB] = conditions["u"]
    [dudxT,dudxB] = conditions["dudx"]
    [dudyT,dudyB] = conditions["dudy"]
    [lapUT,lapUB] = conditions["lap(u)"]
    
    [vT,vB] = conditions["u"]
    [dvdxT,dvdxB] = conditions["dvdx"]
    [dvdyT,dvdyB] = conditions["dvdy"]
    [lapVT,lapVB] = conditions["lap(v)"]
    
    times,_,_,channels = jnp.shape(dataset)
    out = []
    temp = createPaddedMesh(dataset[0][:,:,0],padding)
    for i in range(times):
        temp = []
        for j in range(channels):
            temp.append(createPaddedMesh(dataset[i][:,:,j],padding))
        
        out.append(jnp.dstack([
            channelFlowPadding(temp[0],padding,uT,uB),       # u
            channelFlowPadding(temp[1],padding,dudxT,dudxB), # dudx
            channelFlowPadding(temp[2],padding,dudyT,dudyB), # dudy
            channelFlowPadding(temp[3],padding,lapUT,lapUB), # lap(u)
            
            channelFlowPadding(temp[4],padding,vT,vB),       # v
            channelFlowPadding(temp[5],padding,dvdxT,dvdxB), # dvdx
            channelFlowPadding(temp[6],padding,dvdyT,dvdyB), # dvdy
            channelFlowPadding(temp[7],padding,lapVT,lapVB), # lap(v)
        ]))
    return out