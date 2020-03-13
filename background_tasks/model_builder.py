"""
SCRIPT IMPORTS
"""
import numpy as np
import math, pandas
import matplotlib.image as mpimg
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.optim as optim
import torchvision.models as models

from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import logging
import sys
import os.path
import ntpath
from os import path
from pathlib2 import Path
import model_class

import warnings  
warnings.filterwarnings('ignore')

def vgg16_change_fully_connected_layer(): 
    model.classifier[6].out_features = 10
    for param in model.parameters():
        param.requires_grad = False
    logging.info('All VGG16 layers frozen')
    network = list(model.classifier.children())[:-1]
    network.extend([nn.Linear(4096, 10)])
    model.classifier = nn.Sequential(*network)
    logging.info('New Layer correctly added to VGG16')

def resnet_change_fully_connected_layer(): 
    model.fc.out_features = 10
    for param in model.parameters():
        param.requires_grad = False
    logging.info('All ResNet50 layers frozen')
    network = list(model.fc.children())[:-1]
    network.extend([nn.Linear(2048, 10)])
    model.fc = nn.Sequential(*network)
    logging.info('New Layer correctly added to ResNet50')

def run(model_type, model_container, epochs):
    logging.info('Begin running')
    label_dict = {}
    if(model_container.dataset == 'AVA' or model_container.dataset == '1'):
        label_dict = model_container.get_ava_labels()
        logging.info('Successfully loaded AVA labels')
    else:
        if(not path.exists('Mar13_labeled_images.txt') or not path.exists('Mar13_unlabeled_images.txt')):
            logging.info('labeled_images.txt and unlabeled_images.txt not found')
            model_container.get_xmp_color_class()
        else:
            logging.info('labeled_images.txt and unlabeled_images.txt found')
        label_dict = model_container.get_file_color_class()
    train, test = model_container.build_dataloaders(label_dict)
    
    if model_type == "vgg16":
        vgg16_change_fully_connected_layer()
    else:
        resnet_change_fully_connected_layer()

    model_container.train_data_function(epochs, train, 'N/A')
    # test_data_function(test)

model_name = sys.argv[1]
dataset = sys.argv[2]
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
model_type = sys.argv[5]

logging.basicConfig(filename='logs/' + model_name + '.log', filemode='w', level=logging.DEBUG)

model_name = model_name + '.pt'

if model_type == 'vgg16':
    model = models.vgg16(pretrained=True)
elif model_type == 'resnet':
    model = models.resnet50(pretrained=True)
else:
    logging.info('Invalid model requested: {}'.format(model))
    sys.exit('Invalid Model')

model_container = model_class.ModelBuilder(model, model_name, batch_size, dataset)

run(model_type, model_container, epochs)