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
        self._transform = transform
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

class ModelBuilder:

    def __init__(self, model, model_name, batch_size, dataset):
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.dataset = dataset
        self.limit_num_pictures = False #limit_num_pictures = 2000
        self.rated_indices = []
        self.ratings = []
        self.bad_indices = []
        self.ava_image_path = "/mnt/md0/reynolds/ava-dataset/images/"
        self.missourian_image_path = "/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump"
        self.ava_labels_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"
        self.ava_tags_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/tags.txt"

        if(dataset == 'AVA' or dataset == '1'):
            logging.info('Using AVA Dataset to train')
            self.image_path = self.ava_image_path
        else:
            logging.info('Using Missourian Dataset to train')
            self.image_path = self.missourian_image_path
        

    def get_xmp_color_class(self):
        labels_file = open('labeled_images.txt', 'w')
        none_file = open('unlabeled_images.txt', 'w')

        for i, (root, _, files) in enumerate(os.walk(self.image_path, topdown=True)):
            for name in files:
                with open(os.path.join(root, name), 'rb') as f:
                    img_str = str(f.read())
                    xmp_start = img_str.find('photomechanic:ColorClass')
                    xmp_end = img_str.find('photomechanic:Tagged')
                    if xmp_start != xmp_end and xmp_start != -1:
                        xmp_str = img_str[xmp_start:xmp_end]
                        if xmp_str[26] != '0':
                            labels_file.write(xmp_str[26] + ', ' + str(os.path.join(root, name)) + ', ' + str(i))
                            # self.rated_indices.append(i)
                        else:
                            none_file.write(xmp_str[26] + ', ' + str(os.path.join(root, name)) + ', ' + str(i))
                            # self.bad_indices.append(i)
        
        labels_file.close()
        none_file.close()

    def get_missourian_mapped_val(self, missourian_val):
        if(missourian_val == 1):
            return 10
        elif(missourian_val == 2):
            return 9
        elif(missourian_val == 3):
            return 7
        elif(missourian_val == 4):
            return 6
        elif(missourian_val == 5):
            return 5
        elif(missourian_val == 6):
            return 4
        elif(missourian_val == 7):
            return 2
        elif(missourian_val == 8):
            return 1
        else:
            return 0

    def get_file_color_class(self):
        pic_label_dict = {}
        try:
            labels_file = open("labeled_images.txt", "r")
            none_file = open("unlabeled_images.txt", "r")
        except OSError:
            logging.error('Could not open Missourian image mapping files')
            sys.exit(1)

        for line in labels_file:
            labels_string = line.split(',')
            file_name = (labels_string[1].split('/')[-1]).split('.')[0]
            self.rated_indices.append(int(labels_string[-1]))
            pic_label_dict[file_name] = self.get_missourian_mapped_val(int(labels_string[0]))

        for line in none_file:
            labels_string = line.split(',')
            file_name = (labels_string[1].split('/')[-1]).split('.')[0]
            self.bad_indices.append(int(labels_string[-1]))
            pic_label_dict[file_name] = 0

        logging.info('Successfully loaded info from Missourian Image Files')
        labels_file.close()
        none_file.close()
        return pic_label_dict

    def get_ava_labels(self):
        pic_label_dict = {}
        limit_lines = 1000000
        try:
            f = open(self.ava_labels_file, "r")
            for i, line in enumerate(f):
                if i >= limit_lines:
                    logging.info('Reached the developer specified line limit')
                    break
                line_array = line.split()
                picture_name = line_array[1]
                aesthetic_values = (line_array[2:])[:10]
                for i in range(0, len(aesthetic_values)): 
                    aesthetic_values[i] = int(aesthetic_values[i])
                pic_label_dict[picture_name] = np.asarray(aesthetic_values).argmax()
            logging.info('label dictionary completed')
            return pic_label_dict
        except OSError:
            logging.error('Cannot open AVA label file')
            sys.exit(1)

    def build_dataloaders(self, class_dict):
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

        # load data and apply the transforms on contained pictures
        train_data = AdjustedDataset(self.image_path, class_dict, transform=_transform)
        test_data = AdjustedDataset(self.image_path, class_dict, transform=_transform)
        logging.info('Training and Testing Dataset correctly transformed') 
        
        num_pictures = len(train_data)
        indices = list(range(num_pictures))
        split = int(np.floor(valid_size * num_pictures))
        if(self.dataset == 2):
            train_idx, test_idx = self.rated_indices, self.bad_indices
        else:
            train_idx, test_idx = indices[split:], indices[:split]

        # Define samplers that sample elements randomly without replacement
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # Define data loaders, which allow batching and shuffling the data
        train_loader = torch.utils.data.DataLoader(train_data,
                    sampler=train_sampler, batch_size=1)
        test_loader = torch.utils.data.DataLoader(test_data,
                    sampler=test_sampler, batch_size=1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        logging.info('ResNet50 is running on {}'.format(device))
        return train_loader, test_loader

    def train_data_function(self, epochs, train_loader, prev_model):
        if(prev_model != 'N/A'):
            try:
                self.model.load_state_dict(torch.load('../neural_net/models/' + prev_model))
            except Exception:
                logging.warning('Failed to find {}, model trained off base resnet50'.format(prev_model))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.4, momentum=0.9)

        self.model.train()
        training_loss = 0
        training_accuracy = 0

        for epoch in range(epochs):
            running_loss = 0.0
            num_correct = 0
            try:
                for i, (data, label) in enumerate(train_loader,0):
                    if self.limit_num_pictures:
                        if i > self.limit_num_pictures:
                            break
                    try:
                        logging.info('label for image {} is {}'.format(i, label))
                        label = torch.LongTensor(label)
                        optimizer.zero_grad()
                        output = self.model(data)
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
                (data, label) = train_loader
                logging.error('Error on epoch #{}, train_loader issue with data: {}\nlabel: {}'.format(epoch, data, label))
                torch.save(self.model.state_dict(), self.model_name)
                sys.exit(1)

            training_loss = running_loss/len(train_loader.dataset)
            training_accuracy = 100 * num_correct/len(train_loader.dataset)
            logging.info('training loss: {}\ntraining accuracy: {}'.format(training_loss, training_accuracy))
            try:
                torch.save(self.model.state_dict(), '../neural_net/models/' + self.model_name)
            except Exception:
                logging.error('Unable to save model: {}, saving backup in root dir and exiting program'.format(self.model_name))
                torch.save(self.model.state_dict(), self.model_name)
                sys.exit(1)