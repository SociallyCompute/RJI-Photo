import numpy as np
from PIL import Image
from net import Net

training_data = []
weights = []

def load_data(filename):
    f = open(filename, "r")
    for line in f:
        im = Image.open(line)
        matrix = np.array(im)
        matrix = ((matrix[0] * matrix[1]), matrix[2])
        training_data.append(matrix)
    f.close()

def train_cnn():
    training_data 

def initialize_weights():
    n = Net(training_data[0].shape)
    for x in training_data:
        n.forward(x)