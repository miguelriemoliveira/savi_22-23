import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import nn


class DataVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
         
        # Setup figure
        self.figure = plt.figure(title)
        self.figure.canvas.manager.set_window_title(title)
        self.figure.set_size_inches(4,3)
        plt.suptitle(title)
        plt.legend(loc='best')
        plt.waitforbuttonpress(0.1)

    def draw(self,xs,ys, layer='default', marker='.', markersize=1, color=[0.5,0.5,0.5], alpha=1, label='', x_label='', y_label=''):

        xs,ys = self.toNP(xs,ys) # make sure we have np arrays
        plt.figure(self.title)


        if not layer in self.handles: # first time drawing this layer
            self.handles[layer] = plt.plot(xs, ys, marker, markersize=markersize, 
                                        color=color, alpha=alpha, label=label)
            plt.legend(loc='best')

        else: # use set to edit plot
            plt.setp(self.handles[layer], data=(xs, ys))  # update lm

        plt.xlabel(x_label)    
        plt.ylabel(y_label)    
        plt.draw()

        key = plt.waitforbuttonpress(0.01)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)

    def toNP(self, xs, ys):
        if torch.is_tensor(xs):
            xs = xs.cpu().detach().numpy()

        if torch.is_tensor(ys):
            ys = ys.cpu().detach().numpy()

        return xs,ys


    def recomputeAxesRanges(self):

        plt.figure(self.title)
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        plt.draw()