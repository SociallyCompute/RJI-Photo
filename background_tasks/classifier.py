'''
SCRIPT IMPORTS
'''
#standard ML/Image Processing imports
import numpy as np
import pandas as pd
import math, pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

import logging
import sys

# no one likes irrelevant warnings
import warnings  
warnings.filterwarnings('ignore')

class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
#         print(tuple_with_path)
        return tuple_with_path

"""
TRANSLATE_LABELS
    Input: N/A
    Output: tag_mapping - matches numeric labels to english
    Comments:

"""
def translate_labels():
    tag_mapping = {}
    try:
        f = open(tags_file, "r")
    except OSError:
        logging.error('Unable to open label translation file, exiting program')
        sys.exit(1)
    for i, line in enumerate(f):
        if i >= limit_lines:
            break
        line_array = line.split()
        tag_mapping[int(line_array[0])] = line_array[1]
    tag_mapping[0] = "Miscellaneous"
    return tag_mapping

"""
GET_LABELS
    Input: N/A
    Output: pic_label_dict - dictionary of labels and pictures
    Comments:
        Pulling the numeric labels from the file and mapping them to the correct image
"""
def get_labels():
    pic_label_dict = {}
    try:
        f = open(label_file, "r")
    except OSError:
        logging.error('Unable to open label file, exiting program')
        sys.exit(1)
    for i, line in enumerate(f):
        if i >= limit_lines:
            break
        line_array = line.split()
        picture_name = line_array[1]
        classifications = (line_array[12:])[:-1]
        for i in range(0, len(classifications)): 
            classifications[i] = int(classifications[i])
        pic_label_dict[picture_name] = classifications
    return pic_label_dict

"""
BUILD_DATALOADERS
    Input: N/A
    Output: 
        1. train_loader - training dataloader
        2. test_loader - testing dataloader
    Comments:
        Breaking the data into two sets, one for training and one for testing
"""
def build_dataloaders():
    _transform = transforms.Compose([transforms.ToTensor()])
    data = ImageFolderWithPaths(data_dir, transform=_transform)
    
    # Define our data transforms to get all our images the same size
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    valid_size = 0.2 # percentage of data to use for test set
    test_data = data
    train_data = data
    num_pictures = len(train_data)

    # Shuffle pictures and split training set
    indices = list(range(num_pictures))

    split = int(np.floor(valid_size * num_pictures))
    train_idx, test_idx = indices[split:], indices[:split]#rated_indices, bad_indices

    # Define samplers that sample elements randomly without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Define data loaders, which allow batching and shuffling the data
    train_loader = torch.utils.data.DataLoader(train_data,
                sampler=train_sampler, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_data,
                sampler=test_sampler, batch_size=1)

    # check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Running model on {}'.format(device))
    vgg16.to(device) # loads the model onto the device (CPU or GPU)

    return train_loader, test_loader

"""
CHANGE_FCL
    Input: N/A
    Output: N/A
    Comments:
        Changing the final layer
"""
def change_fcl():
    for param in vgg16.parameters():
        param.requires_grad = False #freeze all convolution weights
    network = list(vgg16.classifier.children())[:-1] #remove fully connected layer
    network.extend([nn.Linear(4096, 67)]) #add new layer of 4096->67
    vgg16.classifier = nn.Sequential(*network)
    logging.info('Added new layer to model')

"""
TRAIN_VGG
    Inputs:
        1. train_loader - dataloader with training data
        2. epochs - how many times we train with the data
        3. pic_label_dict - labels
        4. tag_mapping - mapping the numeric labels to English readable labels
    Output: N/A
    Comments:
        Training function, takes in all the data and trains the model. Eventually saves it off as the model name.
"""
def train_vgg(train_loader, epochs, pic_label_dict, tag_mapping):
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9) # optimizer

    vgg16.train() # set model to training model
    limit_num_pictures = 50
    num_epochs = epochs 
    training_loss = 0
    training_accuracy = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        try:
            for i, data in enumerate(train_loader,0):
                if limit_num_pictures:
                    if i > limit_num_pictures:
                        break
                for j in range(2): #done so we can utilize both labels
                    inputs, _, path = data
                    path = path[0]
                    path_array = path.split('/')
                    pic_name = path_array[-1]
                    label = pic_label_dict[pic_name.split('.')[0]][j]
                    label = torch.LongTensor([label])
                    optimizer.zero_grad()
                    output = vgg16(inputs)
                    loss = criterion(output, label)
                    running_loss += loss.item()
                    _, preds = torch.max(output.data, 1)
                    num_correct += (preds == pic_label_dict[pic_name.split('.')[0]][j]).sum().item()
                    loss.backward()
                    optimizer.step()
                    logging.info('Completed Picture #{}'.format(i))
        except Exception:
            logging.error('Non-specific error in training, dumping dataloader and exiting\n{}'.format(data))
            sys.exit(1)
        logging.info('Trained epoch {}'.format(epoch))
        training_loss = running_loss/len(train_loader.dataset)
        training_accuracy = 100 * num_correct/len(train_loader.dataset)
        logging.info('training loss: {} and training accuracy: {}'.format(training_loss, training_accuracy))

    torch.save(vgg16.state_dict(), '../neural_net/models/CLASSIFICATION_Feb17_All_AVA_only_training.pt')
    torch.save(vgg16.state_dict(), 'BACKUP_CLASSIFICATION_Feb17.pt')

"""
RUN
    Input: number of epochs
    Output: N/A
    Comments:
        Function that is running the entire program
"""
def run(epochs):
    pic_label_dict = get_labels()
    tag_mapping = translate_labels()
    train_loader, test_loader = build_dataloaders()
    change_fcl()
    train_vgg(train_loader, epochs, pic_label_dict, tag_mapping)

"""
GLOBAL VARIABLES
    Vars:
        1. vgg16 - model we are working with
        2. data_dir - location of AVA images
        3. label_file - location of labels
        4. tags_file - location of label translator
        5. limit_lines - limit of lines to read from files
        6. epochs - number of times to run training model through data
"""
logging.basicConfig(filename='logs/classifier.log', filemode='w', level=logging.DEBUG)
# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)

# root directory where the images are stored
data_dir = "/mnt/md0/reynolds/ava-dataset/"
label_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"
tags_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/tags.txt"
limit_lines = 1000000
epochs = 1
run(epochs)
