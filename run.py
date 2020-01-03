#standard ML/Image Processing imports
import numpy as np
import math, pandas
import matplotlib.image as mpimg

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#pytorch imports
import torch
import torch.optim as optim
import torchvision.models as models

from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from metadata.pull_xmp_file import find_color_code
from neural_net.adapted_vgg_classifier_file import run_vgg, load_split_train_test

home = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/Cherryhomes, Ellie"
vgg16 = models.vgg16(pretrained=True)
path_list = []

if __name__ == "__main__":
    training, testing = load_split_train_test(home, .2)
    find_color_code(training)
    run_vgg(training, testing)