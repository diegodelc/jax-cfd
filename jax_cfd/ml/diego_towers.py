import haiku as hk
from jax_cfd.ml import nonlinearities

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
        
        if CNN_specs["nonlinearity"] == "relu":
            nonlinearity = nonlinearities.relu
        
        for i in range(CNN_specs["hidden_layers"]):
            components.append(hk.Conv2D(output_channels=CNN_specs["hidden_channels"], kernel_shape=(3,3), padding="SAME"))
            components.append(nonlinearity)
        
        components.append(hk.Conv2D(output_channels=CNN_specs["num_output_channels"], kernel_shape=(3,3), padding="SAME"))
        
        self.components = components
        
        self.convModel = hk.Sequential(self.components)
        
        
    def __call__(self, x):
        x = self.convModel(x)
        return x

# def ConvNet(x):
#     cnn = CNN(CNN_specs)
#     return cnn(x)


def build_forward_pass(CNN_specs=None):
    return hk.without_apply_rng(hk.transform(ConvNet))