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

import matplotlib.pyplot as plt

import sqlalchemy as sqla

import warnings  
warnings.filterwarnings('ignore')

from common import config
from common import datasets
from common import misc

class ModelBuilder:

    """ ModelBuilder class makes a generalized model. It takes what it is given, 
        creates a dataset and provides testing/training functions for the model.

    :param model_type: (torchvision.model) model type (vgg, resnet, etc)
    :param model_name: (string) name of the model when saving after 
        training or loading for testing
    :param batch_size: (int) batch size, 1 - SGD other is minibatch
    :param dataset: (string) value to identify dataset in usage (ava, missourian)
    """
    def __init__(self, model_type, model_name, batch_size, dataset, classification_subject):
        
        self.model_type = model_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.dataset = dataset
        
        # flag for how many images to load
        self.limit_num_pictures = None
        
        # list to hold indices for images
        self.rated_indices = []
        
        # list of image ratings
        self.ratings = []
        
        # list to hold indices for images with zero labels
        self.bad_indices = []
        
        self.train_data_samples = None
        self.test_data_samples = None
        self.classification_subject = classification_subject
        
        self.db, self.photo_table = helpers.make_db_connection('evaluation')

        if (dataset == 'ava'):
            logging.info('Using AVA Dataset')
            self.image_path = config.AVA_IMAGE_PATH
        else:
            logging.info('Using Missourian Dataset')
            self.image_path = config.MISSOURIAN_IMAGE_PATH

    
    def build_dataloaders(self, class_dict):
        """
        
        :param class_dict: (dict: path->label) dictionary mapping a string path to an int label
        :rtype: ((torch.utils.data.dataloader), (torch.utils.data.dataloader)) training and 
            testing dataloaders (respectively) containing the training images
                test_loader:  dataloader containing the testing images
        """
        _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # percentage of data to use for test set 
        valid_size = 0.2 

        # load data and apply the transforms on contained pictures
        train_data = datasets.AdjustedDataset(self.image_path, class_dict, 
                                     self.dataset, transform=_transform)
        self.train_data_samples = train_data.samples
        
        test_data = datasets.AdjustedDataset(self.image_path, class_dict, self.dataset, transform=_transform)
        self.test_data_samples = test_data.samples
        
        logging.info('Training and Testing Dataset correctly transformed') 
        logging.info('Training size: {}\nTesting size: {}'.format(len(train_data), len(test_data)))
        
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
                    sampler=train_sampler, batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(test_data,
                    sampler=test_sampler, batch_size=self.batch_size)

        logging.info('train_loader size: {}\n'.format(len(train_loader)),
                     'test_loader size: {}'.format(len(test_loader)))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        logging.info('ResNet50 is running on {}'.format(device))
        return train_loader, test_loader

    
    def get_xmp_color_class(self):
        """ Read .txt files from Missourian and write a dictionary mapping an image to a label
        
        :rtype: (dict: path->label) dictionary mapping string path to a specific image to int label
        """
        pic_label_dict = {}
        xmp_db, xmp_table = helpers.make_db_connection('xmp_color_classes')

        xmp_data_SQL = sqla.sql.text("""
        SELECT photo_path, color_class
        FROM xmp_color_classes
        """)
        xmp_data = pandas.read_sql(xmp_data_SQL, xmp_db, params={})
        xmp_data = xmp_data.set_index('photo_path')
        pic_label_dict = {k : v[0] for k, v in xmp_data.to_dict('list').items()}
        
        xmp_rated_index_SQL = sqla.sql.text("""
        SELECT photo_path, os_walk_index
        FROM xmp_color_classes
        WHERE color_class <> 0
        """)
        rated_index_data = pandas.read_sql(xmp_rated_index_SQL, xmp_db, params={})
        rated_index_data = rated_index_data.set_index('photo_path')
        self.rated_indices = rated_index_data['os_walk_index'].to_list()

        xmp_unrated_index_SQL = sqla.sql.text("""
        SELECT photo_path, os_walk_index
        FROM xmp_color_classes
        WHERE color_class = 0
        """)
        unrated_index_data = pandas.read_sql(xmp_unrated_index_SQL, xmp_db, params={})
        unrated_index_data = unrated_index_data.set_index('photo_path')
        self.bad_indices = unrated_index_data['os_walk_index'].to_list()

        logging.info('Successfully loaded info from Missourian Image Files')
        return pic_label_dict

    
    def get_ava_quality_labels(self):
        """ Read .txt files from AVA dataset and write a dictionary mapping an image to a label
        
        :rtype: (dict: path->label) dictionary mapping string path to a specific image to int label
        """
        pic_label_dict = {}
        limit_lines = self.limit_num_pictures

        f = open(self.ava_labels_file, "r")
        for i, line in enumerate(f):
            if limit_lines:
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

    
    def get_ava_content_labels(self):
        """ Similar to get_ava_labels but for classification purposes
        
        :rtype: (dict: path->label) dictionary mapping string path to a specific 
            image to int classification label                
        """
        pic_label_dict = {}
        limit_lines = self.limit_num_pictures
        try:
            f = open(self.ava_labels_file, "r")
            for i, line in enumerate(f):
                if limit_lines:
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


    def evaluate(self, test_loader, num_ratings):
        """ Evaluate the testing set and save them to the database
        
        :param test_loader: (torch.utils.data.dataloader) dataloader containing the testing images
        :param num_ratings: (int) number of labels in the set (i.e. 10 for labels 1-10)
        """
        
        self.model.eval()
        ratings = []
        index_progress = 0
        logging.info('Running evaluation images in the test_loader of size: '
                     '{}...'.format(len(test_loader)))

        while index_progress < num_pictures - 1:
            try:
                for i, (data, labels) in enumerate(test_loader, index_progress):

                    photo_path = self.test_data_samples[index_progress][0]

                    if torch.cuda.is_available():
                        labels = torch.cuda.LongTensor(labels.to('cuda:0'))
                        data = data.to('cuda:0')
                    else:
                        labels = torch.LongTensor(labels)

                    output = self.model(data)

                    # _, preds = torch.max(output.data, 1)
                    ratings = output[0].cpu().tolist()

                    # Prime tuples for database insertion
                    database_tuple = {}
                    for n in range(num_ratings):
                        database_tuple['model_score_{}'.format(n + 1)] = ratings[n]

                    # Include metadata for database tuple
                    database_tuple['photo_path'] = photo_path
                    database_tuple['photo_model'] = self.model_name

                    # Insert tuple to database
                    result = self.db.execute(self.photo_table.insert().values(database_tuple))

            except Exception as e:
                logging.info('Ran into error for image #{}: {}\n... '
                             'Moving on.\n'.format(index_progress, e))
                
            index_progress += 1
            
        logging.info('Finished evaluation of images in the test_loader, the '
                     'results are stored in the photo table in the database')


    def train(self, epochs, train_loader, prev_model):
        """ Train model and save them as pytorch model. Save images 
            showing accuracy and loss functions over epochs.
            
        :param epochs:
        :param test_loader: (torch.utils.data.dataloader) dataloader 
            containing the testing images
        :param prev_model:    
        """
        if(prev_model != 'N/A'):
            try:
                self.model.load_state_dict(torch.load('../neural_net/models/' + prev_model))
            except Exception:
                logging.warning(
                    'Failed to find {}, model trained off base resnet50'.format(prev_model))

        criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # self.model.train()
        training_loss = [0 for i in range(epochs)]
        training_accuracy = [0 for i in range(epochs)]
        
        for epoch in range(epochs):
            running_loss = 0.0
            num_correct = 0
            
            index_progress = 0
            logging.info('Epoch #{} Running the training of images in the '.format(epoch),
                         'train_loader of size: {}...'.format(len(train_loader)))
            
            while index_progress < num_pictures - 1:
                try:
                    for i, (data, label) in enumerate(train_loader,0):
                        if self.limit_num_pictures:
                            if i > self.limit_num_pictures:
                                break
                        try:
                            if torch.cuda.is_available():
                                label = torch.cuda.LongTensor(label.to('cuda:0'))
                                data = data.to('cuda:0')
                                # max_t2 = torch.cuda.LongTensor(1)
                            else:
                                label = torch.LongTensor(label)
                                # max_t2 = 1
                            # logging.info('label for image {} is {}'.format(i, label))
                            optimizer.zero_grad()
                            output = self.model(data)
                            loss = criterion(output, label)
                            running_loss += loss.cpu().item()
                            # _, preds = torch.max(output.data, max_t2)
                            max_vals = torch.max(output.data).long()
                            preds = torch.argmax(output.data).long()
                            num_correct += (preds == label).cpu().sum().item()
                            loss.backward()
                            optimizer.step()
                        except Exception as e:
                            logging.warning('Issue calculating loss and optimizing with image ',
                                            '#{}, error is {}\ndata is\n{}'.format(i, e, data))
                            continue

                        if i % 2000 == 1999:
                            running_loss = 0
                            
                except Exception:
                    (data, label) = train_loader
                    logging.error('Error on epoch #{}, train_loader issue ',
                                  'with data: {}\nlabel: {}'.format(epoch, data, label))
                    torch.save(self.model.state_dict(), self.model_name)
                    sys.exit(1)
                    
                index_progress += 1

            training_loss[epoch] = running_loss/len(train_loader.dataset)
            training_accuracy[epoch] = 100 * num_correct/len(train_loader.dataset)
            logging.info('training loss: {}\ntraining accuracy: {}'.format(
                training_loss[epoch], training_accuracy[epoch]))
            try:
                torch.save(self.model.state_dict(), '../neural_net/models/' + self.model_name)
            except Exception:
                logging.error('Unable to save model: {}, '.format(self.model_name),
                              'saving backup in root dir and exiting program')
                torch.save(self.model.state_dict(), self.model_name)
                sys.exit(1)

        plt.plot([i for i in range(epochs)], training_accuracy)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Training Model Accuracy')
        plt.savefig('graphs/Train_Accuracy_' + self.model_name[:-3] + '.png')

        plt.plot([i for i in range(epochs)], training_loss)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training Model Loss')
        plt.savefig('graphs/Train_Loss_' + self.model_name[:-3] + '.png')
