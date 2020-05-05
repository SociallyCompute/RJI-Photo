'''
SCRIPT IMPORTS
'''
# standard ML/Image Processing imports
import numpy as np
import pandas as pd
import math, pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from PIL import Image

# pytorch imports
import torch
import torch.optim as optim
import torchvision.models as models

from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import sys

# sqlalchemy import for db insertion
import sqlalchemy as s
from sqlalchemy import MetaData
from sqlalchemy.ext.automap import automap_base

import logging

# no one likes irrelevant warnings
import warnings  
warnings.filterwarnings('ignore')

# our custom classes for loading images with paths and/or ratings
from model_class import AdjustedDataset, ModelBuilder


""" Define global variables that are configurable """

# model we will be evaluating with
photo_model = sys.argv[1]
dataset = sys.argv[2]
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
model_type = sys.argv[5]

# setup logger
logging.basicConfig(filename='evaluation_{}.log'.format(photo_model.split('/')[3].split('.')[0]), filemode='w', level=logging.INFO)

# root directory where the images are stored
data_dir = "/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump"

# The number of classification groups
num_ratings = 10

# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
model = models.vgg16(pretrained=True)

model_container = ModelBuilder(model, model_name, batch_size, dataset)

test_loader = build_dataloader(class_dict)
test_data(test_loader, db, photo_table)



""" Database Connection """
DB_STR = 'postgresql://{}:{}@{}:{}/{}'.format(
    'rji', 'donuts', 'nekocase.augurlabs.io', '5433', 'rji'
)
logging.info("Connecting to database: {}".format(DB_STR))

dbschema = 'rji'
db = s.create_engine(DB_STR, poolclass=s.pool.NullPool,
    connect_args={'options': '-csearch_path={}'.format(dbschema)})

# produce our own MetaData object
metadata = MetaData()

# we can reflect it ourselves from a database, using options
# such as 'only' to limit what tables we look at...
metadata.reflect(db, only=['photo'])

# we can then produce a set of mappings from this MetaData.
Base = automap_base(metadata=metadata)

# calling prepare() just sets up mapped classes and relationships.
Base.prepare()

# mapped classes are ready
photo_table = Base.classes['photo'].__table__

logging.info("Database connection successful")



""" Model refining for our needs and loading our pretrained model """
# for param in vgg16.parameters():
#     param.requires_grad = False #freeze all convolution weights

# remove fully connected layer
network = list(model.classifier.children())[:-1] 

# add new layer of 4096->100 (rating scale with 
#     1 decimal - similar to 1 hot encoding)
network.extend([nn.Linear(4096, num_ratings)]) 
model.classifier = nn.Sequential(*network)


""" Load images and create an iterator """
def build_dataloader(class_dict):
    # define our transforms to apply on each image
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_data = AdjustedDataset(data_dir, class_dict, transform=_transform)   

    num_pictures = len(test_data)
    logging.info("Number of pictures in subdirectories: {}".format(num_pictures))

    # Get a list of all numerical indices of the pictures (their numerical order)
    indices = list(range(num_pictures))
    logging.info("Head of indices: {}".format(indices[:10]))

    # Define sampler that sample elements randomly without replacement
    test_sampler = SubsetRandomSampler(indices)

    # Define data loader, which allow batching and shuffling the data
    test_loader = torch.utils.data.DataLoader(test_data,
                sampler=test_sampler, batch_size=1)#, num_workers=4)
    logging.info("Test loader length: {}".format(len(test_loader)))

    # check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Device that will be used: {}".format(device))
    model.to(device) # loads the model onto the device (CPU or GPU)
    model.load_state_dict(torch.load(photo_model))
    logging.info("Successfully loaded our pretrained model: {}".format(model)) #logging.info out the model to ensure our network is correct
    return test_loader

""" Evaluation """
def test_data(test_loader, db, photo_table):
    model.eval() # set model to evaluation/prediction mode
    ratings = []
    index_progress = 0

    while index_progress < len(test_loader) - 1:
        try:
            for i, data in enumerate(test_loader, index_progress):
                
                inputs, _, photo_path = data
                photo_path = photo_path[0]

                output = model(inputs)

                _, preds = torch.max(output.data, 1)
                ratings = output[0].tolist()
                
                logging.info("\nImage photo_path: {}\n".format(photo_path))
                logging.info("Classification for test image #{}: {}\n".format(index_progress, ratings))
                
                # Prime tuples for database insertion
                database_tuple = {}
                for n in range(num_ratings):
                    database_tuple['model_score_{}'.format(n + 1)] = ratings[n]

                # Include metadata for database tuple
                database_tuple['photo_path'] = photo_path
                database_tuple['photo_model'] = photo_model
                logging.info("Tuple to insert to database: {}\n".format(database_tuple))

                # Insert tuple to database
                result = db.execute(photo_table.insert().values(database_tuple))
                logging.info("Primary key inserted into the photo table: {}\n".format(result.inserted_primary_key))

                index_progress += 1

        except Exception as e:
            logging.info("Ran into error for image #{}: {}\n... Moving on.\n".format(index_progress, e))
            index_progress += 1

