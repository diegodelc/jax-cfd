import haiku as hk
from jax_cfd.ml import nonlinearities
from jax_cfd.ml.towers import scale_to_range
import functools
import jax.numpy as jnp
from jax.random import PRNGKey as randomKey

class CNN(hk.Module):
    def __init__(self,CNN_specs=None):
        if CNN_specs == None:
            CNN_specs = {
                "hidden_channels" : 16,
                "hidden_layers" : 2,
                "nonlinearity" : "relu",
                "num_output_channels" : 8
            }
        super().__init__(name="CNN")
        components = []
        
#         components.append(functools.partial(scale_to_range, axes=ndim_axes))
        
        if CNN_specs["nonlinearity"] == "relu":
            nonlinearity = nonlinearities.relu
        
        for i in range(CNN_specs["hidden_layers"]):
            components.append(hk.Conv2D(output_channels=CNN_specs["hidden_channels"], kernel_shape=(3,3), padding="SAME"))
            components.append(nonlinearity)
#             components.append(hk.dropout(randomKey(42),0.2))
        
        components.append(hk.Conv2D(output_channels=CNN_specs["num_output_channels"], kernel_shape=(3,3), padding="SAME"))
#         ndim = 2
#         self.ndim_axes = list(range(ndim))
#         self.scale_fn = functools.partial(scale_to_range, axes=self.ndim_axes)
        
    
        self.components = components
        
        self.convModel = hk.Sequential(self.components)
        
        
    def __call__(self, x):
#         minx = jnp.min(x)
#         maxx = jnp.max(x)
#         meanX = jnp.mean(x)
#         stdX = jnp.std(x)
#         normX = (x-meanX)/stdX
#         x = self.convModel(normX)
#         return x*stdX + meanX
        x = self.convModel(x)
        return x
#         return self.convModel(scale_to_range(x,self.ndim_axes,0,1))

class CNN_Dropout(hk.Module):
    def __init__(self, CNN_specs):
        super().__init__(name="CNN")
        self.CNN_specs = CNN_specs
        
        
    def __call__(self,x):
        for i in range(self.CNN_specs["hidden_layers"]):
            x = hk.Conv2D(output_channels=self.CNN_specs["hidden_channels"], kernel_shape=(3,3), padding="SAME")(x)
            x = nonlinearities.relu(x)
            x = hk.dropout(randomKey(42),0.2,x)
        
        x = hk.Conv2D(output_channels=self.CNN_specs["num_output_channels"], kernel_shape=(3,3), padding="SAME")(x)
        return x


def build_forward_pass(CNN_specs=None):
    return hk.without_apply_rng(hk.transform(ConvNet))