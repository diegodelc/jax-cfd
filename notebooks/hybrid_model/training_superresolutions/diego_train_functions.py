import numpy as np
import jax.numpy as jnp

import jax
from jax import value_and_grad

import time


def printAllShapes(*things):
    for thing in things:
        print(np.shape(thing))

        
# Loss functions
def MeanSquaredErrorLoss(params, input_data, actual,forward_pass):
    preds,truth = computePredAndTruth(params,input_data,actual,forward_pass)
    return jnp.power(jnp.array(truth) - jnp.array(preds), 2).mean()

def MeanSquaredErrorLossWithDivergence(params, input_data, actual):
    preds,truth = computePredAndTruth(params,input_data,actual)
    mse = jnp.power(jnp.array(truth) - jnp.array(preds), 2).mean()
    div = divergence(v0)
    h = 1
    return mse + h * div

def absErrorLoss(params,input_data,actual):
    preds,truth = computePredAndTruth(params,input_data,actual)
    return ((jnp.array(truth) - jnp.array(preds))).mean()

#reference:
# https://goodboychan.github.io/python/deep_learning/vision/tensorflow-keras/2020/10/13/01-Super-Resolution-CNN.html#Build-SR-CNN-Model
def mse(target, ref):
    target_data = target.astype(np.float32)
    ref_data = ref.astype(np.float32)
    err = np.sum((target_data - ref_data) ** 2)
    
    err /= float(target_data.shape[0] * target_data.shape[1])
    return err

def my_mse(pred,actual):
    err = (jnp.array(pred)-jnp.array(actual))**2
    return err.mean()

def percentageError(approx, truth):
    return (approx-truth)/truth

def absPercentageError(approx,truth):
    return abs(percentageError(approx,truth))

def mape(approx, truth):
    return jnp.mean(abs(percentageError(approx,truth)))





# Training functions
def staggeredLearningRate(*args):
    """
    accepts tuples with the following info:
    (number of epochs, learning rate)
    
    example usage: 
    learning_rates = staggeredLearningRate((50,0.005),(50,0.001))
    """
    learning_rates = []
    for arg in args:
        for i in range(arg[0]):
            learning_rates.append(arg[1])
    return learning_rates


def computePredAndTruth(params,input_data,actual,forward_pass):
    preds = []
    truth = []
    for i in range(len(input_data)):
        preds.append(forward_pass.apply(params, input_data[i]))
        truth.append(actual[i])
    return preds,truth


def train_step(params, X_train, Y_train,X_val,Y_val,learning_rate,forward_pass):
    loss, param_grads = value_and_grad(MeanSquaredErrorLoss)(params, X_train, Y_train,forward_pass)
    val_loss = MeanSquaredErrorLoss(params, X_val,Y_val,forward_pass)
    return jax.tree_map(lambda p,g: p-learning_rate*g, params, param_grads), loss, val_loss


def UpdateWeights(weights,gradients,learning_rate):
    return weights - learning_rate * gradients


def train(X_train,Y_train,X_test,Y_test,rng_key,input_channels,epochs,learning_rates,printEvery=5,params=None,forward_pass=None,tol = 1e-5):
    """
    Input parameter 'params' allows us to keep training a network that has already undergone some 
    training, without having to retrain from scratch
    """
    
    shapes = np.shape(X_train)
    if input_channels != shapes[3]:
        raise(AssertionError("Non-compatible input shape and input channels"))

    sample_x = jax.random.uniform(rng_key, (shapes[1],shapes[2],input_channels))

    batch_size = 1
    if params == None:
        params = forward_pass.init(rng_key, sample_x)

    
    print("Shapes of all datasets")
    printAllShapes(X_train,Y_train,X_test,Y_test)
    print("\n")
    
    
    losses = []
    val_losses = []
    start_time = time.time()
    for i in range(1, epochs+1):
        
        
        #will fail if learning rates list is not long enough, in which case it keeps using value from last loop
        learning_rate = learning_rates[i-1]     
        
        
        
        params,loss,val_loss = train_step(params,X_train,Y_train,X_test,Y_test,learning_rate,forward_pass) #TODO: using test as validation, change this!

        
        if i%printEvery == 0: #every n epochs
            print("Epoch {:.0f}/{:.0f}".format(i,epochs))
            print("\tmse : {:.6f}\t".format(loss), end='')
            print("\tval mse : {:.6f}".format(val_loss), end='')
            time_now = time.time()
            time_taken = time_now - start_time
            time_per_epoch = time_taken/i
            epochs_remaining = epochs+1-i
            time_remaining = epochs_remaining*time_per_epoch
            end_time = time.localtime(time_now + time_remaining)
            print("\tEstimated end time: {:d}:{:02d}:{:02d}".format(end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
            print("\n")
        losses.append(loss)
        val_losses.append(val_loss)

        if i != 1:
            if abs(losses[-2]-losses[-1])<tol:
                print("\nConvergence reached at epoch {:.0f}".format(i))
                print("\tmse : {:.6f}\t".format(loss), end='')
                print("\tval mse : {:.6f}".format(val_loss), end='')
                print("\n")
                break

    if i == epochs:
        print("\nFinished training at max epochs\n")
    
    return losses, val_losses, params