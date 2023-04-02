import haiku as hk
class newSaveObject():
    def __init__(self,params,forward_pass):
        self.params = params
        self.forward_pass = forward_pass

        
        
class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        components = []
        
        components.append(hk.Conv2D(output_channels=2*num_output_channels, kernel_shape=(3,3), padding="SAME"))
        components.append(nonlinearities.relu)
        components.append(hk.Conv2D(output_channels=2*num_output_channels, kernel_shape=(3,3), padding="SAME"))
        components.append(nonlinearities.relu)
        components.append(hk.Conv2D(output_channels=num_output_channels, kernel_shape=(3,3), padding="SAME"))
        components.append(nonlinearities.relu)
        
        self.components = components

    def __call__(self, x):
        x = hk.Sequential(self.components)(x)
        return x

def ConvNet(x):
    cnn = CNN()
    return cnn(x)

