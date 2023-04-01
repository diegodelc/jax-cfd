import haiku as hk


def forward_pass_module(
    num_output_channels,
    ndim,
    tower_module
):
    """Constructs a function that initializes tower and applies it to inputs."""
    global forward_pass ## this is to make the forward_pass_module pickleable
    def forward_pass(inputs):
        return tower_module(num_output_channels, ndim)(inputs)

    return forward_pass

class SaveObject():
#     def __init__(self,params):
#         self.params = params
    
    def save_params(self,params):
        self.params = params
    
    def save_forward_pass_params(self,num_output_channels,ndim,tower_module):
        self.num_output_channels = num_output_channels
        self.ndim = ndim
        self.tower_module = tower_module
    
    def make_forward_pass(self):
        self.forward_pass = forward_pass_module(num_output_channels = self.num_output_channels, 
                                                    ndim = self.ndim,
                                                    tower_module = self.tower_module)
        
    def preprocess(self):
        self.forward_pass = hk.without_apply_rng(hk.transform(self.forward_pass))
