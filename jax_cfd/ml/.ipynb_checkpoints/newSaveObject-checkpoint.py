import haiku as hk

class newSaveObject():
    def __init__(self,params,losses,val_losses,description,CNN_specs=None):
        self.params = params
        self.CNN_specs = CNN_specs
        self.losses = losses
        self.val_losses = val_losses
        self.description = description
