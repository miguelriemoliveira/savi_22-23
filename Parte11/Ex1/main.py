#!/usr/bin/env python3

# -----------------------------------------------------------------
# Project: SAVI 2022-2023
# Author: Miguel Riem Oliveira
# Inspired in:
# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
# -----------------------------------------------------------------

import argparse
import pickle
from copy import deepcopy

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
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


def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    # Read file with points
    file = open('pts.pkl', 'rb')
    pts = pickle.load(file)
    file.close()
    print('pts = ' + str(pts))

    # Convert the pts into np arrays
    xs_np = np.array(pts['xs'], dtype=np.float32).reshape(-1,1)
    ys_np_labels = np.array(pts['ys'], dtype=np.float32).reshape(-1,1)

    # Draw training data
    plt.plot(xs_np, ys_np_labels,'g.', label = 'labels')
    plt.legend(loc='best')
    plt.show()

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu


    model = Model() # Instantiate model
    model.to(device) # move the model variable to the gpu if one exists

    learning_rate = 0.01
    maximum_num_epochs = 50 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
  
    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    idx_epoch = 0
    while True:

        xs_ten = torch.from_numpy(xs_np).to(device)
        ys_ten_labels = torch.from_numpy(ys_np_labels).to(device)

        # Apply the network to get the predicted ys
        ys_ten_predicted = model.forward(xs_ten)

        # Compute the error based on the predictions
        loss = criterion(ys_ten_predicted, ys_ten_labels)

        # Update the model, i.e. the neural network's weights 
        optimizer.zero_grad() # resets the weights to make sure we are not accumulating
        loss.backward() # propagates the loss error into each neuron
        optimizer.step() # update the weights

        # Report
        print('Epoch ' + str(idx_epoch) + ', Loss ' + str(loss.item()))

        idx_epoch += 1 # go to next epoch
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    # Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(xs_ten)
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    plt.plot(xs_np, ys_np_labels,'g.', label = 'labels')
    plt.plot(xs_np, ys_np_predicted,'rx', label = 'predicted')
    plt.legend(loc='best')
    plt.show()


    

if __name__ == "__main__":
    main()

