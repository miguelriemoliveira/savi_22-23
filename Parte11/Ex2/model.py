
import torch


# Definition of the model. For now a 1 neuron network
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__() # call superclass constructor        

        # Define the structure of the neural network
        self.layer1 = torch.nn.Linear(1,1)

    def forward(self, xs):
        
        ys = self.layer1(xs)
        
        return ys
