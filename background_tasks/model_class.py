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

import sqlalchemy as s
from sqlalchemy import MetaData
from sqlalchemy.ext.automap import automap_base

import matplotlib.pyplot as plt

import warnings  
warnings.filterwarnings('ignore')

"""
AdjustedDataset
    Input: DatasetFolder object

    Attributes: 
        transform: (Tensor Object) Description of how to transform an image.
        classes: (list) List of the class names.
        class_to_idx: (dict) pairs of (image, class).
        samples: (list) List of (sample_path, class_index) tuples.
        targets: (list) class_index value for each image in dataset.

    Methods:
        __init__
        __getitem__
        __len__
        _find_classes
        get_class_dict
        make_dataset
        pil_loader
        

    Comments:
        Based on this - https://discuss.pytorch.org/t/custom-label-for-torchvision-imagefolder-class/52300/8
        Adjusts dataset to prep for dataloader, map labels to images
"""
class AdjustedDataset(datasets.DatasetFolder):
    """
    __init__
        Input:
            image_path: (string) Folder where the image samples are kept.
            class_dict: (dict) pairs of (picture, rating)
            transform: (Object) Image processing transformations.
    """
    def __init__(self, image_path, class_dict, dataset, transform=None):
        self.target_transform = None
        self.transform = transform
        self.dataset = dataset
        self.classes = [i+1 for i in range(10)] #classes are 1-10
        self.class_to_idx = {i+1 : i for i in range(10)}
        self.samples = self.make_dataset(image_path, class_dict)
        self.targets = [s[1] for s in self.samples]

    """
    __getitem__
        Input:
            index: (int) Item index

        Output:
            tuple: (tensor, int) where target is class_index of target_class. Same as DatasetFolder object
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.pil_loader(path) #transform Image into Tensor
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    """
    __len__
        Comments:
            DEPRECATED USAGE
    """
    def __len__(self):
        return len(self.samples)

    """
    _find_classes
        Comments:
            DEPRECATED USAGE
    """
    def _find_classes(self, class_dict):
        classes = list(class_dict.values())
        class_to_idx = {classes[i] : i for i in range(len(classes))}
        return classes, class_to_idx

    """
    get_class_dict
        Comments:
            DEPRECATED USAGE
    """
    def get_class_dict(self):
        return self.class_to_idx

    """
    make_dataset
        Input:
            1. root: (string) root path to images
            2. class_to_idx: (dict: string, int) image name, mapped to class
        
        Output:
            images: [(path, class)] list of paths and mapped classes
    """
    def make_dataset(self, root, class_to_idx):
        images = []
        root_path = os.path.expanduser(root)
        for r, _, fnames in os.walk(root_path):
            for fname in sorted(fnames):
                path = os.path.join(r, fname)
                logging.info('path is {}'.format(path))
                # AVA dataset has lowercase, Missourian usable pictures are uppercase, unusable are lowercase
                if (path.lower().endswith(('.png', '.jpg')) and self.dataset == 'AVA') or (path.endswith('.JPG')):
                    item = (path, class_to_idx[fname.split('.')[0]])
                    logging.info('appending item {}'.format(item))
                    images.append(item)

        return images

    """
    pil_loader
        Input:
            full_path: (string) path to image for loading

        Output:
            RGB PIL image type
    """
    def pil_loader(self, full_path):
        image = Image.open(full_path)
        image = image.convert('RGB')
        return image


"""
ModelBuilder
    Attributes:
        model: (torchvision.model) specify the model to run
        model_name: (string) name of the model when saving after training or loading for testing
        batch_size: (int) batch size, 1 - SGD other is minibatch
        dataset: (int) value to identify dataset in usage
        limit_num_pictures: (boolean or int) flag for how many images to load
        rated_indices: (list: int) list to hold indices for images
        ratings: (list: int) list of image ratings
        bad_indices: (list: int) list to hold indices for images with zero labels
        ava_image_path: (string) path to ava images
        missourian_image_path: (string) path to Missourian images
        ava_labels_file: (string) path to ava labels
        ava_tags_file: (string) path to ava label to tag mappings
        db: (sqlalchemy engine) reference to database engine
        photo_table: (sqlalchemy table) table

    Methods:
        __init__
        build_dataloaders
        get_xmp_color_class
        get_missourian_mapped_val
        get_file_color_class
        get_ava_labels
        get_classifier_labels
        make_db_connection
        test_data_function
        train_data_function

    Comments:
        ModelBuilder class makes a generalized model. It takes what it is given, creates a dataset and 
        provides testing/training functions for the model.
"""
class ModelBuilder:

    """
    __init__
        Input:
            model: (torchvision.model) model type
            model_name: (string) name to save or load model
            batch_size: (int) size of batch
            dataset: (int) identify which dataset to use
        Output:
    """
    def __init__(self, model, model_name, batch_size, dataset):
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.dataset = dataset
        self.limit_num_pictures = False #limit_num_pictures = 2000
        self.rated_indices = []
        self.ratings = []
        self.bad_indices = []
        self.train_data_samples = None
        self.test_data_samples = None
        self.ava_image_path = "/storage/hpc/group/augurlabs/images/"
        # self.ava_image_path = "/mnt/md0/reynolds/ava-dataset/images/"
        self.missourian_image_path = "/storage/hpc/group/augurlabs/2016/Fall/Dump"
        # self.missourian_image_path = "/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump"
        self.ava_labels_file = "/storage/hpc/group/augurlabs/ava/AVA.txt"
        # self.ava_labels_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"
        self.ava_tags_file = "/storage/hpc/group/augurlabs/ava/tags.txt"
        # self.ava_tags_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/tags.txt"
        self.db, self.photo_table = self.make_db_connection()

        if(dataset == 'AVA' or dataset == '1'):
            logging.info('Using AVA Dataset to train')
            self.image_path = self.ava_image_path
        else:
            logging.info('Using Missourian Dataset to train')
            self.image_path = self.missourian_image_path

    """
    build_dataloaders
        Input:
            class_dict: (dict: path->label) dictionary mapping a string path to an int label
        Output:
            train_loader: (torch.utils.data.dataloader) dataloader containing the training images
            test_loader: (torch.utils.data.dataloader) dataloader containing the testing images
    """
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
        train_data = AdjustedDataset(self.image_path, class_dict, self.dataset, transform=_transform)
        self.train_data_samples = train_data.samples
        test_data = AdjustedDataset(self.image_path, class_dict, self.dataset, transform=_transform)
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

        logging.info('train_loader size: {}\ntest_loader size: {}'.format(len(train_loader), len(test_loader)))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        logging.info('ResNet50 is running on {}'.format(device))
        return train_loader, test_loader
        
    """
    get_xmp_color_class
        Input:
            N/A
        Output:
            N/A
        Comments:
            write string containing Missourian labels to .txt files
    """
    def get_xmp_color_class(self):
        labels_file = open('Mar18_labeled_images.txt', 'w')
        none_file = open('Mar18_unlabeled_images.txt', 'w')
        i = 0

        for root, _, files in os.walk(self.image_path, topdown=True):
            for name in files:
                if name.endswith('.JPG', '.PNG'):
                    with open(os.path.join(root, name), 'rb') as f:
                        img_str = str(f.read())
                        xmp_start = img_str.find('photomechanic:ColorClass')
                        xmp_end = img_str.find('photomechanic:Tagged')
                        if xmp_start != xmp_end and xmp_start != -1:
                            xmp_str = img_str[xmp_start:xmp_end]
                            if xmp_str[26] != '0':
                                labels_file.write(xmp_str[26] + '; ' + str(os.path.join(root, name)) + '; ' + str(i) + '\n')
                            else:
                                none_file.write(xmp_str[26] + '; ' + str(os.path.join(root, name)) + '; ' + str(i) + '\n')
                        else:
                            none_file.write('0; ' + str(os.path.join(root, name)) + '; ' + str(i) + '\n')
                        i+=1
        
        labels_file.close()
        none_file.close()

    """
    get_missourian_mapped_val
        Input:
            missourian_val: (int) value read from Missourian XMP metadata
        Output:
            (int) converted Missourian Value
        Comments:
            Simple helper function to convert Missourian XMP Meta labels to standardized labels
    """
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

    """
    get_file_color_class
        Input:
            N/A
        Output:
            pic_label_dict: (dict: path->label) dictionary mapping string path to a specific image to int label
        Comments:
            Read .txt files from Missourian and write a dictionary mapping an image to a label
    """
    def get_file_color_class(self):
        pic_label_dict = {}
        try:
            labels_file = open("Mar18_labeled_images.txt", "r")
            none_file = open("Mar18_unlabeled_images.txt", "r")
        except OSError:
            logging.error('Could not open Missourian image mapping files')
            sys.exit(1)

        for line in labels_file:
            labels_string = line.split(';')
            file_name = (labels_string[1].split('/')[-1]).split('.')[0]
            self.rated_indices.append(int(labels_string[-1].split('/')[0]))
            pic_label_dict[file_name] = self.get_missourian_mapped_val(int(labels_string[0]))

        for line in none_file:
            labels_string = line.split(';')
            file_name = (labels_string[1].split('/')[-1]).split('.')[0]
            self.bad_indices.append(int(labels_string[-1].split('/')[0]))
            pic_label_dict[file_name] = 0

        logging.info('Successfully loaded info from Missourian Image Files')
        labels_file.close()
        none_file.close()
        return pic_label_dict

    """
    get_ava_labels
        Input:
            N/A
        Output:
            pic_label_dict: (dict: path->label) dictionary mapping string path to a specific image to int label
        Comments:
            similarly to get_file_color_class
    """
    def get_ava_labels(self):
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
                aesthetic_values = (line_array[2:])[:10]
                for i in range(0, len(aesthetic_values)): 
                    aesthetic_values[i] = int(aesthetic_values[i])
                pic_label_dict[picture_name] = np.asarray(aesthetic_values).argmax()
            logging.info('label dictionary completed')
            return pic_label_dict
        except OSError:
            logging.error('Cannot open AVA label file')
            sys.exit(1)

    """
    get_classifier_labels
        Input:
            N/A
        Output:
            pic_label_dict: (dict: path->label) dictionary mapping string path to a specific image to int classification label
        Comments:
            Similar to get_ava_labels but for classification purposes
    """
    def get_classifier_labels(self):
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

    """
    make_db_connection
        Input:
            N/A
        Output:
            db: (sqlalchemy engine) reference to database engine
            photo_table: (sqlalchemy table) table 
        Comments:
            Makes a connection to the database used to store each of the testing values. Allows for 
            standardization of test values to recieve a decent test result
    """
    def make_db_connection(self):
        DB_STR = 'postgresql://{}:{}@{}:{}/{}'.format(
            'rji', 'donuts', 'nekocase.augurlabs.io', '5433', 'rji'
        )
        logging.info("Connecting to database: {}".format(DB_STR))

        dbschema = 'rji'
        db = s.create_engine(DB_STR, poolclass=s.pool.NullPool,
            connect_args={'options': '-csearch_path={}'.format(dbschema)})
        metadata = MetaData()
        metadata.reflect(db, only=['photo'])
        Base = automap_base(metadata=metadata)
        Base.prepare()
        photo_table = Base.classes['photo'].__table__

        logging.info("Database connection successful")
        return db, photo_table

    """
    test_data_function
        Input:
            test_loader: (torch.utils.data.dataloader) dataloader containing the testing images
            num_ratings: (int) number of labels in the set (i.e. 10 for labels 1-10)
        Output:
            N/A
        Comments:
            test results and save them to the database
    """
    def test_data_function(self, test_loader, num_ratings):
        self.model.eval()
        ratings = []
        index_progress = 0
        logging.info('Test_loader is size: {}'.format(len(test_loader)))
        # while index_progress < len(test_loader) - 1:
        
        for data, labels in test_loader:
            try:
                # inputs, _, photo_path = data
                # photo_path = photo_path[0]
                photo_path = self.test_data_samples[index_progress][0]

                output = self.model(data)

                _, preds = torch.max(output.data, 1)
                ratings = output[0].tolist()
                
                # logging.info("\nImage photo_path: {}\n".format(photo_path))
                logging.info("Classification for test image #{}: {}\n".format(index_progress, ratings))
                
                # Prime tuples for database insertion
                database_tuple = {}
                for n in range(num_ratings):
                    database_tuple['model_score_{}'.format(n + 1)] = ratings[n]

                # Include metadata for database tuple
                database_tuple['photo_path'] = photo_path
                database_tuple['photo_model'] = self.model_name
                logging.info("Tuple to insert to database: {}\n".format(database_tuple))

                # Insert tuple to database
                result = self.db.execute(self.photo_table.insert().values(database_tuple))
                logging.info("Primary key inserted into the photo table: {}\n".format(result.inserted_primary_key))

                index_progress += 1

            except Exception as e:
                logging.info("Ran into error for image #{}: {}\n... Moving on.\n".format(index_progress, e))
                index_progress += 1

    """
    train_data_function:
        Input:
            test_loader: (torch.utils.data.dataloader) dataloader containing the testing images
        Output:
            N/A
        Comments:
            Train model and save them as pytorch model. Save images showing accuracy and loss functions over epochs.
    """
    def train_data_function(self, epochs, train_loader, prev_model):
        if(prev_model != 'N/A'):
            try:
                self.model.load_state_dict(torch.load('../neural_net/models/' + prev_model))
            except Exception:
                logging.warning('Failed to find {}, model trained off base resnet50'.format(prev_model))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.4, momentum=0.9)

        self.model.train()
        training_loss = [0 for i in range(epochs)]
        training_accuracy = [0 for i in range(epochs)]

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
                        if torch.cuda.is_available():
                            try:
                                # label = torch.cuda.LongTensor(label)
                                label = torch.cuda.LongTensor(label.to('cuda:0'))
                                data = data.to('cuda:0')
                                max_t2 = torch.cuda.LongTensor(1)
                            except Exception as e:
                                logging.error('Error with cuda version. Error {}'.format(e))
                                sys.exit(1)
                        else:
                            torch.LongTensor(label)
                            max_t2 = 1
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, label)
                        running_loss += loss.item()
                        _, preds = torch.max(output.data, max_t2)
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

            training_loss[epoch] = running_loss/len(train_loader.dataset)
            training_accuracy[epoch] = 100 * num_correct/len(train_loader.dataset)
            logging.info('training loss: {}\ntraining accuracy: {}'.format(training_loss[epoch], training_accuracy[epoch]))
            try:
                torch.save(self.model.state_dict(), '../neural_net/models/' + self.model_name)
            except Exception:
                logging.error('Unable to save model: {}, saving backup in root dir and exiting program'.format(self.model_name))
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