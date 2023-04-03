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
    conditions = [
            [0,0],#u
            [0,0],#dudx
            [0,0],#dudy
            [0,0],#lapu

            [0,0],#v
            [0,0],#dvdx
            [0,0],#dvdy
            [0,0]#lapv
        ]
    """
    times,_,_,channels = jnp.shape(dataset)
    
    if conditions is None:
        conditions = []
        for i in range(channels):
            conditions.append([0,0])
    
    
    
    if channels != len(conditions):
        raise AssertionError("Number of channels and bcs are incompatible")
    out = []
    temp = createPaddedMesh(dataset[0][:,:,0],padding)
    for i in range(times):
        temp_out = []
        for j in range(channels):
            temp_out.append(channelFlowPadding(
                                createPaddedMesh(dataset[i][:,:,j],padding),
                                padding,
                                conditions[j][0],
                                conditions[j][1],
                            )
            )
        
        out.append(jnp.dstack(temp_out))
        
    return out