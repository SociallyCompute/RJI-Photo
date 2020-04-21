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
from common import model

import warnings  
warnings.filterwarnings('ignore')


def vgg16_change_fully_connected_layer(output_layer): 
    """ Change fully connected layer for vgg16 and apply a mapping from 4096->output_layer
    
    :param output_layer: (int) number of output layers on final fully connected layer
    """
    logging.info('Initial VGG16 Architecture: {}'.format(list(model_active.classifier.children())))
    model_active.classifier[6].out_features = output_layer
    for param in model_active.parameters():
        param.requires_grad = False
    logging.info('All VGG16 layers frozen')
    network = list(model_active.classifier.children())[:-1]
    network.extend([nn.Linear(4096, output_layer)])
    model_active.classifier = nn.Sequential(*network)
    logging.info('Changed VGG16 Architecture: {}'.format(list(model_active.classifier.children())))
    # logging.info('New Layer correctly added to VGG16')

"""
resnet_change_fully_connected_layer
    Input:
        output_layer: (int) number of output layers on final fully connected layer
    Output:
        N/A
    Comments:
        Change fully connected layer for resnet and apply a mapping from 2048->output_layer
"""
def resnet_change_fully_connected_layer(output_layer): 
    logging.info('Initial ResNet50 final layer Architecture: {}'.format(model_active.fc))
    model_active.fc.out_features = output_layer
    for param in model_active.parameters():
        param.requires_grad = False
    logging.info('All ResNet50 layers frozen')
    model_active.fc = nn.Linear(2048, output_layer)
    logging.info('Changed ResNet50 Architecture: {}'.format(model_active.fc))
    # logging.info('New Layer correctly added to ResNet50')

"""
run_train_model
    Input:
        model_type: (string) identify which type of model is being run
        model_container: (ModelBuilder) custom ModelBuilder base class
        epochs: (int) total number of times to train the model
        output_layer: (int) number of output layers on final fully connected layer
    Output:
        N/A
    Comments:
        Basic run function for training models
"""
def run_train_model(model_type, model_container, epochs, output_layer):
    # AVA
    if(model_container.dataset == 'ava'): 
        label_df = model_container.get_ava_quality_labels()

    # Missourian
    elif(model_container.dataset == 'missourian'): 
        if(not path.exists('Mar13_labeled_images.txt') or \
           not path.exists('Mar13_unlabeled_images.txt')):
            logging.info('labeled_images.txt or unlabeled_images.txt not found')
            model_container.get_xmp_color_class()
        else:
            logging.info('labeled_images.txt and unlabeled_images.txt found')
        label_dict = model_container.get_file_color_class()
    else: #classifier
        label_dict = model_container.get_classifier_labels()

    train, _ = model_container.build_dataloaders(label_dict)

    if model_type == "vgg16":
        vgg16_change_fully_connected_layer(output_layer)
    else:
        resnet_change_fully_connected_layer(output_layer)

    model_container.train_data_function(epochs, train, 'N/A')


def run_test_model(model_type, model_container, output_layer):
    """ Test models
    
    :param model_type: (string) identify which type of model is being run
    :param model_container: (ModelBuilder) custom ModelBuilder base class
    :param output_layer: (int) number of output layers on final fully connected layer
    """
    logging.info('Begin running')
    label_dict = {}
    label_dict = model_container.get_file_color_class()
    _, test = model_container.build_dataloaders(label_dict)
    if model_type == 'vgg16':
        vgg16_change_fully_connected_layer(output_layer)
    else:
        resnet_change_fully_connected_layer(output_layer)

    model_container.test_data_function(test, output_layer)

model_name = sys.argv[1]
dataset = sys.argv[2]
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])

# 'vgg16' or 'resnet'
model_type = sys.argv[5] 

# 'content' or 'quality'
classification_subject = sys.argv[6]

logging.basicConfig(filename='logs/' + model_name + '.log', 
                    filemode='w', level=logging.DEBUG)
model_name = model_name + '.pt'

if classification_subject == 'quality':
    output_layer = 10
elif classification_subject == 'content':
    output_layer = 67
else:
    logging.info('The classification subject you specified ({}) '.format(classification_subject)
                'does not exist, please choose from \'quality\' or \'content\'\n')
    sys.exit(1)

if model_type == 'vgg16':
    model_active = models.vgg16(pretrained=True).cuda() if torch.cuda.is_avaliable() else models.vgg16(pretrained=True)
elif model_type == 'resnet':
    model_active = models.resnet50(pretrained=True).cuda() if torch.cuda.is_available() else models.resnet50(pretrained=True)
else:
    logging.info('Invalid model requested: {}. '.format(model_active)
                'Please choose from \'vgg16\' or \'resnet\'\n')
    sys.exit('Invalid Model')

model_container = model.ModelBuilder(model_active, model_name, batch_size, dataset, classification_subject)

if os.path.isfile('../neural_net/models/' + model_name):
    logging.info('Running Model in Testing Mode')
    run_test_model(model_type, model_container, output_layer)
else:
    logging.info('Running Model in Training Mode')
    run_train_model(model_type, model_container, epochs, output_layer)
    





