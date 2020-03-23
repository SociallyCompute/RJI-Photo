'''
SCRIPT IMPORTS
'''
import numpy as np
import math, pandas
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

import warnings  
warnings.filterwarnings('ignore')

"""
AdjustedDataset
    Input: DatasetFolder object
    __init__: 
        root to images (string), image to class dict (dict: string, int), transform operator(transform Object), max class (int)
    Comments:
        expects 1 class per image. Needs to be adjusted to incorporate 2+ classes per image
"""
# https://discuss.pytorch.org/t/custom-label-for-torchvision-imagefolder-class/52300/8
class AdjustedDataset(datasets.DatasetFolder):
    def __init__(self, image_path, class_dict, transform=None):
        """Imports dataset from folder structure.

        Input:
            image_path: (string) Folder where the image samples are kept.
            class_dict: (dict) pairs of (picture, rating)
            transform: (Object) Image processing transformations.

        Attributes: 
            classes: (list) List of the class names.
            class_to_idx: (dict) pairs of (image, class).
            samples: (list) List of (sample_path, class_index) tuples.
            targets: (list) class_index value for each image in dataset.
        """
        #super(AdjustedDataset, self).__init__(image_path, self.pil_loader, extensions=('.jpg', '.png', '.PNG', '.JPG'),transform=transform)
        self.target_transform = None
        self.transform = transform
        self.classes = [i+1 for i in range(10)] #classes are 1-10
        self.class_to_idx = {i+1 : i for i in range(10)}
        # self.classes, self.class_to_idx = self._find_classes(class_dict)
        self.samples = self.make_dataset(image_path, class_dict)
        self.targets = [s[1] for s in self.samples]

    def pil_loader(self, full_path):
        image = Image.open(full_path)
        image = image.convert('RGB')
        #tensor_sample = transforms.ToTensor()(image)
        return image

    def _find_classes(self, class_dict):
        classes = list(class_dict.values())
        class_to_idx = {classes[i] : i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, root, class_to_idx):
        '''
        Input:
            1. root: (string) root path to images
            2. class_to_idx: (dict: string, int) image name, mapped to class
        
        Output:
            images: [(path, class)] list of paths and mapped classes
        '''
        images = []
        root_path = os.path.expanduser(root)
        for r, _, fnames in os.walk(root_path):
            for fname in sorted(fnames):
                path = os.path.join(r, fname)
                logging.info('path is {}'.format(path))
                if path.lower().endswith(('.png', '.jpg')):
                    item = (path, class_to_idx[fname.split('.')[0]])
                    logging.info('appending item {}'.format(item))
                    images.append(item)

        return images


    def get_class_dict(self):
        """Returns a dictionary of classes mapped to indicies."""
        return self.class_to_idx

    def __getitem__(self, index):
        """Returns tuple: (tensor, int) where target is class_index of
        target_class. Same as DatasetFolder object
        
        Args:
            idx: (int) Index.
        """

        path, target = self.samples[index]
        sample = self.pil_loader(path) #transform Image into Tensor
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

"""
TRANSLATE_LABELS
    Input: N/A
    Output: tag_mapping - matches numeric labels to english
    Comments:

"""
def translate_labels():
    tag_mapping = {}
    limit_lines = 1000000
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
    limit_lines = 1000000
    try:
        f = open(label_file, "r")
        for i, line in enumerate(f):
            if i >= limit_lines:
                logging.info('Reached the developer specified line limit')
                break
            line_array = line.split()
            picture_name = line_array[1]
            classifications = (line_array[12:])[:-1]
            for i in range(0, len(classifications)): 
                classifications[i] = int(classifications[i])
            pic_label_dict[picture_name] = classifications
        logging.info('label dictionary completed')
        return pic_label_dict
    except OSError:
        logging.error('Unable to open label file, exiting program')
        sys.exit(1)

"""
BUILD_DATALOADERS
    Input: N/A
    Output: 
        1. train_loader - training dataloader
        2. test_loader - testing dataloader
    Comments:
        Breaking the data into two sets, one for training and one for testing
"""
def build_dataloaders(label_dict):
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # load data and apply the transforms on contained pictures
    train_data = AdjustedDataset(data_dir, label_dict, transform=_transform)
    test_data = AdjustedDataset(data_dir, label_dict, transform=_transform)
    logging.info('Training and Testing Dataset correctly transformed') 

    valid_size = 0.2 # percentage of data to use for test set

    num_pictures = len(train_data)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vgg16.to(device)
    logging.info('Running model on {}'.format(device))
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
def train_vgg(train_loader, epochs, pic_label_dict, tag_mapping, model_name):
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9) # optimizer

    vgg16.train() # set model to training model
    training_loss = 0
    training_accuracy = 0
    num_epochs = epochs 

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        try:
            for i, (data, labels) in enumerate(train_loader,0):
                if limit_num_pictures:
                    if i > limit_num_pictures:
                        break
                
                try:
                    logging.info('labels for image {} is {} and {}'.format(i, labels[0], labels[1]))
                    for i in range(0, 2):
                        label = torch.LongTensor(labels[i])
                        optimizer.zero_grad()
                        output = vgg16(data)
                        loss = criterion(output, label)
                        running_loss += loss.item()
                        _, preds = torch.max(output.data, 1)
                        num_correct += (preds == label).sum().item()
                        loss.backward()
                        optimizer.step()
                except Exception:
                    logging.warning('Issue calculating loss and optimizing with image #{}, data is\n{}'.format(i, data))
                    continue
                
                if i % 2000 == 1999:
                    running_loss = 0

        except Exception:
            (data, labels) = train_loader
            logging.error('Error on epoch #{}, train_loader issue with data: {}\nlabels: {} and {}'.format(epoch, data, labels[0], labels[1]))
            torch.save(vgg16.state_dict(), 'Backup_classifier_model.pt')
            sys.exit(1)

        training_loss = running_loss/len(train_loader.dataset)
        training_accuracy = 100 * num_correct/len(train_loader.dataset)
        logging.info('training loss: {}\ntraining accuracy: {}'.format(training_loss, training_accuracy))
        try:
            torch.save(vgg16.state_dict(), '../neural_net/models/' + model_name)
        except Exception:
            logging.error('Unable to save model: {}, saving backup in root dir and exiting program'.format(model_name))
            torch.save(vgg16.state_dict(), 'Backup_classifier_model.pt')
            sys.exit(1)


"""
RUN
    Input: number of epochs
    Output: N/A
    Comments:
        Function that is running the entire program
"""
def run():
    logging.info('Begin running')
    pic_label_dict = get_labels()
    tag_mapping = translate_labels()
    train_loader, test_loader = build_dataloaders(pic_label_dict)
    change_fcl()
    train_vgg(train_loader, epochs, pic_label_dict, tag_mapping, model_name)

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

model_name = sys.argv[1]
if(model_name.split('.')[1] != 'pt'):
    logging.info('Invalid model name {} submitted, must end in .pt or .pth'.format(model_name))
    sys.exit('Invalid Model')
epochs = 1

# root directory where the images are stored
data_dir = "/mnt/md0/reynolds/ava-dataset/"
label_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"
tags_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/tags.txt"
limit_num_pictures = False # limit_num_pictures = 1000000
vgg16 = models.vgg16(pretrained=True)

run()
