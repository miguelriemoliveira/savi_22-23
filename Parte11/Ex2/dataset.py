
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, num_points, m ,b, sigma=1.4):

        self.num_points = num_points
        
        # Generate data
        self.xs_np = np.random.rand(num_points,1) * 20 - 10 # to get values between -10 and 10
        self.xs_np = self.xs_np.astype(np.float32)
        self.ys_np_labels = m * self.xs_np + b # compute ys 
        self.ys_np_labels += np.random.normal(loc=0.0, scale=sigma, size=(num_points,1)) # add noise

        # convert to torch tensor
        self.xs_ten = torch.from_numpy(self.xs_np)
        self.ys_ten = torch.from_numpy(self.ys_np_labels)

    def __getitem__(self, index): # return a specific element x,y given the index, of the dataset
        return self.xs_ten[index], self.ys_ten[index]
    
    def __len__(self): # return the length of the dataset
        return self.num_points