import numpy as np
from PIL import Image
import torch

def load_data(filename):
    f = open(filename, "r")
    for line in f:
        im = Image.open(line)
    f.close()

def train_cnn():

def initialize_weights():