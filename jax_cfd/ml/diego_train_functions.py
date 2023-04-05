import numpy as np
import jax.numpy as jnp

import jax
from jax import value_and_grad

import jax_cfd.base as cfd
from jax_cfd.base import grids

import time

def printAllShapes(*things):
    for thing in things:
        print(np.shape(thing))

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


        
class MyTraining():
    def __init__(self,X_train,Y_train,X_test,Y_test,rng_key,input_channels,epochs,learning_rates,printEvery=5,params=None,forward_pass=None,tol = 1e-5):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.rng_key = rng_key
        self.input_channels = input_channels
        self.epochs = epochs
       
        self.printEvery = printEvery
        self.params = params
        self.forward_pass = forward_pass
        self.tol = tol
        
        self.losses = []
        self.val_losses = []
        
        self.learning_rates = learning_rates
        self.learning_rate = self.learning_rates[0]
    
    
    # Loss functions
    def MeanSquaredErrorLoss(self, params,input_data, actual):
        preds,truth = self.computePredAndTruth(params,input_data,actual)
        
        return jnp.power(jnp.array(truth) - jnp.array(preds), 2).mean()
    # MeanSquaredErrorLoss = jax.jit(MeanSquaredErrorLoss)


#     def MeanSquaredErrorLossWithDivergence(self,params, input_data, actual):
#         preds,truth = self.computePredAndTruth(input_data,actual)
#         mse = jnp.power(jnp.array(truth) - jnp.array(preds), 2).mean()
#         div = divergence(v0)
#         h = 1
#         return mse + h * div

    def absErrorLoss(self,input_data,actual):
        preds,truth = computePredAndTruth(input_data,actual)
        return ((jnp.array(truth) - jnp.array(preds))).mean()

    #reference:
    # https://goodboychan.github.io/python/deep_learning/vision/tensorflow-keras/2020/10/13/01-Super-Resolution-CNN.html#Build-SR-CNN-Model
    def mse(self,target, ref):
        target_data = target.astype(np.float32)
        ref_data = ref.astype(np.float32)
        err = np.sum((target_data - ref_data) ** 2)

        err /= float(target_data.shape[0] * target_data.shape[1])
        return err

    def my_mse(self,pred,actual):
        err = (jnp.array(pred)-jnp.array(actual))**2
        return err.mean()

    def percentageError(self,approx, truth):
        return (approx-truth)/truth

    def absPercentageError(self,approx,truth):
        return abs(percentageError(approx,truth))

    def mape(self,approx, truth):
        return jnp.mean(abs(percentageError(approx,truth)))





    # Training functions



    def computePredAndTruth(self,params,input_data,actual):
        preds = []
        truth = []
        for i in range(len(input_data)):
            preds.append(self.forward_pass.apply(params, input_data[i]))
            truth.append(actual[i])

        return preds,truth
    
    


    def train_step(self):
        loss, param_grads = value_and_grad(self.MeanSquaredErrorLoss)(self.params,self.X_train, self.Y_train)
        val_loss = self.MeanSquaredErrorLoss(self.params,self.X_test,self.Y_test)
        
#         return jax.tree_map(self.UpdateWeights, param_grads), loss, val_loss
        return jax.tree_map(lambda p,g: p-self.learning_rate*g, self.params, param_grads), loss, val_loss

#     train_step_jit = jax.jit(train_step)

    def UpdateWeights(self,gradients):
        return self.params - self.learning_rate * gradients

    

    
    def train(self):
        """
        Input parameter 'params' allows us to keep training a network that has already undergone some 
        training, without having to retrain from scratch
        """

        shapes = np.shape(self.X_train)
        if self.input_channels != shapes[3]:
            print(self.input_channels,shapes[3])
            raise(AssertionError("Non-compatible input shape and input channels"))

        sample_x = jax.random.uniform(self.rng_key, (shapes[1],shapes[2],self.input_channels))

        batch_size = 1
        if self.params == None:
            self.params = self.forward_pass.init(self.rng_key, sample_x)


        print("Shapes of all datasets")
        printAllShapes(self.X_train,self.Y_train,self.X_test,self.Y_test)
        print("\n")


        
        start_time = time.time()
        start_time_local = time.localtime(start_time)
        print("\nStart time: {:d}:{:02d}:{:02d}".format(start_time_local.tm_hour, start_time_local.tm_min, start_time_local.tm_sec))
        
        
        for i in range(1, self.epochs+1):


            #will fail if learning rates list is not long enough, in which case it keeps using value from last loop
            self.learning_rate = self.learning_rates[i-1]     


            #TODO: using test as validation, change this!
            self.params,loss,val_loss = self.train_step()
            



            if i%self.printEvery == 0 or i == 1: #every n epochs
                print("Epoch {:.0f}/{:.0f}".format(i,self.epochs))
                print("\tmse : {:.6f}\t".format(loss), end='')
                print("\tval mse : {:.6f}".format(val_loss), end='')
                time_now = time.time()
                time_taken = time_now - start_time
                time_per_epoch = time_taken/(i+1)
                epochs_remaining = self.epochs+1-(i+1)
                time_remaining = epochs_remaining*time_per_epoch
                end_time = time.localtime(time_now + time_remaining)
                print("\tEstimated end time: {:d}:{:02d}:{:02d}".format(end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
                print("\n")
            self.losses.append(loss)
            self.val_losses.append(val_loss)

            if i != 1:
                if abs(self.losses[-2]-self.losses[-1])<self.tol:
                    print("\nConvergence reached at epoch {:.0f}".format(i))
                    print("\tmse : {:.6f}\t".format(loss), end='')
                    print("\tval mse : {:.6f}".format(val_loss), end='')
                    print("\n")
                    break

        if i == self.epochs:
            print("\nFinished training at max epochs\n")

    
        



def reshapeData(mydata,mygrid,
                offsets=[(1.0,0.5),
                         (0,5,1.0)],
                bcs = [cfd.boundaries.channel_flow_boundary_conditions(ndim=2),
                      cfd.boundaries.channel_flow_boundary_conditions(ndim=2)]):
    """defaults to channel flow settings"""
    return (grids.GridVariable(array = grids.GridArray(data = mydata[:,:,0],offset=offsets[0],grid=mygrid),bc=bcs[0]),
          grids.GridVariable(array = grids.GridArray(data = mydata[:,:,1],offset=offsets[1],grid=mygrid),bc=bcs[1]))