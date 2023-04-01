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
    axis=1 -> top and bottom (transpose then left and right then transpose)
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

def channelFlowPadding(data,kernel,topWall,lowWall):
    (padRow,padCol) = findPadding(kernel)
    if padRow != padCol:
        raise AssertionError('Only square filters supported')
    
    data = periodicPadding(data,padRow,axis = 1)
    
    
    data = dirichletPadding(data,padCol,topWall,lowWall,axis = 0)
    
    return padCorners(data,padRow,0)

def retrieveField(data,kernel):
    (padRow,padCol) = findPadding(kernel)
    return data.at[padRow:-padRow,padCol:-padCol].get()