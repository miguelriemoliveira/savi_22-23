import random

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import nn
from torchvision import datasets, models, transforms


class ClassificationVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

    def draw(self, inputs, labels, outputs):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(8,6)
        plt.suptitle(self.title)
        plt.legend(loc='best')

        inputs = inputs
        batch_size,_,_,_ = list(inputs.shape)

        output_probabilities = F.softmax(outputs, dim=1).tolist()
        output_probabilities_dog = [x[0] for x in output_probabilities]

        random_idxs = random.sample(list(range(batch_size)), k=5*5)
        for plot_idx, image_idx in enumerate(random_idxs, start=1):

            label = labels[image_idx]
            output_probability_dog = output_probabilities_dog[image_idx]

            is_dog = True if output_probability_dog > 0.5 else False
            success = True if (label.data.item() == 0 and is_dog) or (label.data.item() == 1 and not is_dog) else False

            image_t = inputs[image_idx,:,:,:]
            image_pil = self.tensor_to_pil_image(image_t)

            ax = self.figure.add_subplot(5,5,plot_idx) # define a 5 x 5 subplot matrix
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            color = 'green' if success else 'red' 
            title = 'dog' if is_dog else 'cat'
            title += ' ' + str(image_idx)
            ax.set_xlabel(title, color=color)

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)

