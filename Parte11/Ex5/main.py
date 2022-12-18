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
from statistics import mean

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import Model
from dataset import Dataset
from colorama import Fore, Style


def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    # Create the dataset
    dataset_train = Dataset(3000, 0.9, 14, sigma=3)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)

    dataset_test = Dataset(500, 0.9, 14, sigma=3)
    # loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)

    # for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):
    #     print('batch ' + str(batch_idx) + ' has xs of size ' + str(xs_ten.shape) )
    # exit(0)

    # Draw training data
    # plt.plot(dataset.xs_np, dataset.ys_np_labels,'g.', label = 'labels')
    # plt.legend(loc='best')
    # plt.show()

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model = Model() # Instantiate model
    model.to(device) # move the model variable to the gpu if one exists

    learning_rate = 0.01
    maximum_num_epochs = 500 
    termination_loss_threshold =  10
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    idx_epoch = 0
    epoch_losses = []
    while True:

        # Train batch by batch
        losses = []
        for batch_idx, (xs_ten, ys_ten_labels) in tqdm(enumerate(loader_train), total=len(loader_train), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):

            xs_ten = xs_ten.to(device)
            ys_ten_labels = ys_ten_labels.to(device)

            # Apply the network to get the predicted ys
            ys_ten_predicted = model.forward(xs_ten)

            # Compute the error based on the predictions
            loss = criterion(ys_ten_predicted, ys_ten_labels)

            # Update the model, i.e. the neural network's weights 
            optimizer.zero_grad() # resets the weights to make sure we are not accumulating
            loss.backward() # propagates the loss error into each neuron
            optimizer.step() # update the weights

            # Report
            # print('Epoch ' + str(idx_epoch) + ' batch ' + str(batch_idx) + ', Loss ' + str(loss.item()))

            losses.append(loss.data.item())

        # Compute the loss for the epoch
        epoch_loss = mean(losses)
        epoch_losses.append(epoch_loss)

        print(Fore.BLUE + 'Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_loss) + Style.RESET_ALL)

        idx_epoch += 1 # go to next epoch
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break
        elif epoch_loss < termination_loss_threshold:
            print('Finished training. Reached target loss.')
            break
            

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    # Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(dataset_train.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    plt.title('Train dataset data')
    plt.plot(dataset_train.xs_np, dataset_train.ys_np_labels,'g.', label = 'labels')
    plt.plot(dataset_train.xs_np, ys_np_predicted,'rx', label = 'predicted')
    plt.legend(loc='best')


    # Plot the loss epoch graph
    plt.figure()
    plt.title('Training report')
    plt.plot(range(0, len(epoch_losses)), epoch_losses,'-b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.draw()


    # Plot the dataset test data and model predictions
    plt.figure()
    ys_ten_predicted = model.forward(dataset_test.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    plt.title('Test dataset data')
    plt.plot(dataset_test.xs_np, dataset_test.ys_np_labels,'g.', label = 'labels')
    plt.plot(dataset_test.xs_np, ys_np_predicted,'rx', label = 'predicted')
    plt.legend(loc='best')



    plt.show()


    

if __name__ == "__main__":
    main()

