import numpy as np
from PIL import Image
import torch
import torch.nn.functional as func
from torch.nn import Module

class Net(Module):

    def __init__(self, shape):
        # possible shapes include:
        #   4912 x 7360 x 3
        #   4016 x 6016 x 3
        #   2008 x 3008 x 3

        super(Net, self).__init__()
        self.first_conv = torch.nn.Conv2d(shape[2], 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.reduction1 = torch.nn.Linear(18 * (shape[0]/2) * (shape[1]/2), 64)
        self.reduction2 = torch.nn.Linear(64, 10) #10 is the number of possible solutions
    
    def forward(self, data):
        #convert to a x y x 18
        data = func.relu(self.first_conv(data))
        #convert to (a/2) x (y/2) x 18
        data = self.pool(data)
        sh = data.shape
        #convert shape to (-1, ((a/2) x (y/2) x 18))
        data = data.view(-1, (sh[0] * sh[1] * sh[2]))
        data = func.relu(self.reduction1(data))
        data = self.reduction2(data)
        return data

    def error(self):
        return torch.nn.CrossEntropyLoss()
