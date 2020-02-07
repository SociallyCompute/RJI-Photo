'''
SCRIPT IMPORTS
'''
#standard ML/Image Processing imports
import numpy as np
import pandas as pd
import math, pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from PIL import Image

#pytorch imports
import torch
import torch.optim as optim
import torchvision.models as models

from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

#sqlalchemy import for db insertion
import sqlalchemy as s
from sqlalchemy import MetaData
from sqlalchemy.ext.automap import automap_base

# no one likes irrelevant warnings
import warnings  
warnings.filterwarnings('ignore')

# our custom classes for loading images with paths and/or ratings
#from .neural_net.image_classification_file import ImageFolderWithPathsAndRatings, ImageFolderWithPaths

""" Database Connection """
DB_STR = 'postgresql://{}:{}@{}:{}/{}'.format(
    'rji', 'hotdog', 'mudcats.augurlabs.io', '5433', 'rji'
)
print("Connecting to database: {}".format(DB_STR))

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

print("Database connection successful")

""" Model and image loading preparation """
# root directory where the images are stored
data_dir = "/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump"
limit_num_pictures = 2000

# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device that will be used: {}".format(device))
vgg16.to(device, '\n') # loads the model onto the device (CPU or GPU)

""" Model refining for our needs and loading our pretrained model """
for param in vgg16.parameters():
    param.requires_grad = False #freeze all convolution weights
network = list(vgg16.classifier.children())[:-1] #remove fully connected layer
network.extend([nn.Linear(4096, 8)]) #add new layer of 4096->100 (rating scale with 1 decimal - similar to 1 hot encoding)
vgg16.classifier = nn.Sequential(*network)

# criterion = nn.CrossEntropyLoss() # loss function
# optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9) # optimizer

vgg16.load_state_dict(torch.load('models/Jan16_All_2017_Fall_Dump_only_labels.pt'))

print("Successfully loaded our pretrained model: {}".format(vgg16)) #print out the model to ensure our network is correct

""" Load images and create an iterator """

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

test_data = ImageFolderWithPaths(data_dir, transform=_transform)   

num_pictures = len(test_data)
print("Number of pictures in subdirectories: {}".format(num_pictures))

# Get a list of all numerical indices of the pictures (their numerical order)
indices = list(range(num_pictures))
print("Head of indices: {}".format(indices[:10]))

# Define sampler that sample elements randomly without replacement
test_sampler = SubsetRandomSampler(indices)

# Define data loader, which allow batching and shuffling the data
test_loader = torch.utils.data.DataLoader(test_data,
            sampler=test_sampler, batch_size=1)#, num_workers=4)
print("Test loader length: {}".format(len(test_loader)))

""" Evaluation """
vgg16.eval() # set model to evaluation/prediction mode
ratings = []
ratings_data = None
for i, data in enumerate(test_loader, 0):
    
    if limit_num_pictures:
        if i > limit_num_pictures:
            break
    inputs, _, path = data
    path = path[0]

    output = vgg16(inputs)

    _, preds = torch.max(output.data, 1)
    ratings = output[0].tolist()
    
    print("\nImage path: {}".format(path))
    print("Classification for test image #{}: {}".format(i, ratings))
    
    tuple_to_insert = {}
    for n in range(8):
        tuple_to_insert[str(n + 1)] = [ratings[n]]
    tuple_to_insert['file_path'] = [path]
    tuple_to_insert = pandas.DataFrame.from_dict(tuple_to_insert)

    if i == 0:
        ratings_data = tuple_to_insert
    else:
        ratings_data = ratings_data.append(tuple_to_insert, ignore_index=True)
    
    display(ratings_data.tail(1))
    
    if i % 100 == 0:
        fig = plt.figure(figsize=(16, 4))
        columns = 3
        rows = 1
        img = mpimg.imread(path)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

ratings_data = ratings_data.set_index('file_path')

""" Standardization of results """
scaler = StandardScaler() 
space_1 = scaler.fit_transform(ratings_data)
ratings_data_norm = pd.DataFrame(space_1, columns=ratings_data.columns, index=ratings_data.index)
ratings_data_norm.hist()
ratings_data.hist()

