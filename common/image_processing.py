import logging, os.path, pandas, sys, torch, warnings

import numpy as np
import sqlalchemy as sqla
import math
from statistics import median
import operator

from os import path
from PIL import Image, ImageFile
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from skimage import feature

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
sys.path.append(os.path.split(sys.path[0])[0])

from common import config
from common import datasets
from common import connections
from common import model

def get_xmp_color_class(rated_indicies, bad_indices):
    """ Read .txt files from Missourian and write a dictionary mapping an image to a label
    
    :rtype: (dict: path->label) dictionary mapping string path to a specific image to int label
    """
    pic_label_dict = {}
    xmp_db, xmp_table = connections.make_db_connection('xmp_color_classes')

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
    rated_indices = rated_index_data['os_walk_index'].to_list()

    xmp_unrated_index_SQL = sqla.sql.text("""
    SELECT photo_path, os_walk_index
    FROM xmp_color_classes
    WHERE color_class = 0
    """)
    unrated_index_data = pandas.read_sql(xmp_unrated_index_SQL, xmp_db, params={})
    unrated_index_data = unrated_index_data.set_index('photo_path')
    bad_indices = unrated_index_data['os_walk_index'].to_list()

    logging.info('Successfully loaded info from Missourian Image Files')
    return pic_label_dict


def get_ava_quality_labels(limit_num_pictures):
    """ Read .txt files from AVA dataset and write a dictionary mapping an image to a label
    
    :rtype: (dict: path->label) dictionary mapping string path to a specific image to int label
    """
    pic_label_dict = {}
    mu = [0, 0, 0, 0, 0, 0, 0, 0]
    std_dev = [0, 0, 0, 0, 0, 0, 0, 0]

    f = open(config.AVA_QUALITY_LABELS_FILE, "r")
    for i, line in enumerate(f):
        if limit_num_pictures:
            if i >= limit_num_pictures:
                logging.info('Reached the developer specified line limit')
                break
        line_array = line.split()
        picture_name = line_array[1]
        aesthetic_values = [int(i) for i in ((line_array[2:])[:10])]
        for i in range(0, 8): #8 is chosen because we have 8 groupings (1,2,3)/(2,3,4)/.../(8,9,10) 
            total = (aesthetic_values[i]*1) + (aesthetic_values[i+1]*2) + (aesthetic_values[i+2]*3)
            n = (aesthetic_values[i]) + (aesthetic_values[i+1]) + (aesthetic_values[i+2])
            #print(aesthetic_values[i])
            #print(aesthetic_values[i+1])
            #print(aesthetic_values[i+2])
            if n != 0:
                mu[i] = (total/n)
                sum_n = (((1-mu[i])**2) * aesthetic_values[i]) + (((2-mu[i])**2) * aesthetic_values[i+1]) + (((3-mu[i])**2) * aesthetic_values[i+2])
                std_dev[i] = math.sqrt(sum_n/n)
            else:
                mu[i] = 0
                std_dev[i] = 0
            
            # aesthetic_values[i] = int(aesthetic_values[i])
        # pic_label_dict[picture_name] = np.asarray(aesthetic_values).argmax()
        med = median(std_dev)
        red_mu = [mu[i] if std_dev[i] <= med else 0 for i in range(0, len(std_dev))]
        index, value = max(enumerate(red_mu), key=operator.itemgetter(1))
        # pic_label_dict[picture_name] = np.mean(np.asarray(aesthetic_values))
        pic_label_dict[picture_name] = np.asarray(value + index)
    # logging.info('label dictionary completed')
    return pic_label_dict

def build_dataloaders(model, class_dict):
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
    valid_size = 0.5 

    # load data and apply the transforms on contained pictures
    train_data = datasets.AdjustedDataset(model.image_path, class_dict, 
                                    model.dataset, model.subject, model.device, transform=_transform)
    model.train_data_samples = train_data.samples
    
    test_data = datasets.AdjustedDataset(model.image_path, class_dict, 
                                model.dataset, model.subject, model.device, transform=_transform)
    model.test_data_samples = test_data.samples
        
    num_pictures = len(train_data)
    indices = list(range(num_pictures))
    split = int(np.floor(valid_size * num_pictures))
    
    if(model.dataset == 2):
        train_idx, test_idx = model.rated_indices, model.bad_indices
    else:
        train_idx, test_idx = indices[split:], indices[:split]

    # Define samplers that sample elements randomly without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Define data loaders, which allow batching and shuffling the data
    train_loader = torch.utils.data.DataLoader(train_data,
                sampler=train_sampler, batch_size=model.batch)
    test_loader = torch.utils.data.DataLoader(test_data,
                sampler=test_sampler, batch_size=model.batch)

    logging.info('train_loader size: {}'.format(len(train_loader)))
    logging.info('test_loader size: {}'.format(len(test_loader)))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.model.to(model.device)
    logging.info('ResNet50 is running on {}'.format(model.device))
    return train_loader, test_loader

def canny_edge_detection(image, sigma=None):
    gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) if image.ndim == 3 else image
    return features.canny(gray_image)
