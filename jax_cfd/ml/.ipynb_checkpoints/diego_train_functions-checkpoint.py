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
    def __init__(self,X_train,Y_train,X_test,Y_test,rng_key,input_channels,epochs,learning_rates,batch_size = 32,validateEvery=1,printEvery=5,params=None,forward_pass=None,tol = 1e-5,PINN_bcs = False,PINN_coeff = 1.0):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.rng_key = rng_key
        self.input_channels = input_channels
        self.epochs = epochs
        
        
        self.batch_size = batch_size
        self.validateEvery = validateEvery
       
        self.printEvery = printEvery
        self.params = params
        self.forward_pass = forward_pass
        self.tol = tol
        
        self.losses = []
        self.val_losses = []
        
        self.learning_rates = learning_rates
        self.learning_rate = self.learning_rates[0]
        
        self.PINN_bcs = PINN_bcs
        self.PINN_coeff = jnp.array(PINN_coeff)
    
    
    # Loss functions
    def MeanSquaredErrorLoss(self, params,input_data, actual,PINN_bcs=False):
        preds,truth = self.computePredAndTruth(params,input_data,actual)
        
        out = jnp.power(jnp.array(truth) - jnp.array(preds), 2).mean()
        
        if PINN_bcs:
            PINN_loss = 0.0
            for thisOne in range(len(preds)):
                
                
               
                for whichVel in [0,1]:
                    
                    
                    PINN_loss += jnp.sum(jnp.abs(preds[thisOne][0,:,whichVel])**2) # top row, all columns
                    PINN_loss += jnp.sum(jnp.abs(preds[thisOne][-1,:,whichVel])**2) # bottom row, all columns
    
                    PINN_loss += jnp.sum(jnp.abs(preds[thisOne][1:-1,0,whichVel])**2) # first column, all rows except for corners
                    PINN_loss += jnp.sum(jnp.abs(preds[thisOne][1:-1,-1,whichVel])**2) # last column, all rows except for corners
                    
            
            out += self.PINN_coeff * PINN_loss
            
            #convert to jnp.array()
        
        
        return out
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

    def train_stepNoValidation(self,X_train_batch,Y_train_batch):
            loss, param_grads = value_and_grad(self.MeanSquaredErrorLoss)(self.params,self.X_train, self.Y_train,PINN_bcs = self.PINN_bcs)


            return jax.tree_map(self.UpdateWeights,self.params,param_grads), loss
#             return jax.tree_map(lambda p,g: p-self.learning_rate*g, self.params, param_grads), loss

    def eval_validation(self):
        return self.MeanSquaredErrorLoss(self.params,self.X_test,self.Y_test)

    def UpdateWeights(self,params,gradients):
        return params - self.learning_rate * gradients

    

    
    def train(self):
        """
        Input parameter 'params' allows us to keep training a network that has already undergone some 
        training, without having to retrain from scratch
        """
        if self.batch_size == None:
            self.batch_size = len(X_train)
        
            
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
        
        num_batches = len(self.X_train)//self.batch_size+1
        
        
        for i in range(1, self.epochs+1):
            if i%self.printEvery == 0 or i == 1: #every n epochs
                print("Epoch {:.0f}/{:.0f}".format(i,self.epochs))

            #will fail if learning rates list is not long enough, in which case it keeps using value from last loop
            self.learning_rate = self.learning_rates[i-1]     


            #TODO: using test as validation, change this!
            for batch in range(num_batches):
                if batch != num_batches-1:
                    start, end = int(batch*self.batch_size), int(batch*self.batch_size+self.batch_size)
                else:
                    start, end = int(batch*self.batch_size), None
                
                X_batch, Y_batch = self.X_train[start:end], self.Y_train[start:end]
                
                self.params,loss = self.train_stepNoValidation(X_batch,Y_batch)
                self.losses.append(loss)
            
            if i%self.validateEvery == 0 or i == 1:
                val_loss = self.eval_validation()
                self.val_losses.append(val_loss)


            if i%self.printEvery == 0 or i == 1: #every n epochs
                
                print("\tmse : {:.6f}\t".format(self.losses[-1]), end='')
                if len(self.val_losses) > 0:
                    print("\tval mse : {:.6f}".format(self.val_losses[-1]), end='')
                time_now = time.time()
                time_taken = time_now - start_time
                time_per_epoch = time_taken/(i+1)
                epochs_remaining = self.epochs+1-(i+1)
                time_remaining = epochs_remaining*time_per_epoch
                end_time = time.localtime(time_now + time_remaining)
                print("\tEstimated end time: {:d}:{:02d}:{:02d}".format(end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
                print("\n")
            
            

            if i != 1:
                if abs(self.losses[-2]-self.losses[-1])<self.tol:
                    print("\nConvergence reached at epoch {:.0f}".format(i))
                    print("\tmse : {:.6f}\t".format(self.losses[-1]), end='')
                    if self.val_loss[-1] is not None:
                        print("\tval mse : {:.6f}".format(self.val_loss[-1]), end='')
                    print("\n")
                    break

        if i == self.epochs:
            print("\nFinished training at max epochs\n")

    
        



def reshapeData(mydata,mygrid,
                offsets=[(1.0,0.5),
                         (0.5,1.0)],
                bcs = [cfd.boundaries.channel_flow_boundary_conditions(ndim=2),
                      cfd.boundaries.channel_flow_boundary_conditions(ndim=2)]):
    """defaults to channel flow settings"""
    return (grids.GridVariable(array = grids.GridArray(data = mydata[:,:,0],offset=offsets[0],grid=mygrid),bc=bcs[0]),
          grids.GridVariable(array = grids.GridArray(data = mydata[:,:,1],offset=offsets[1],grid=mygrid),bc=bcs[1]))