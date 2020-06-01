import numpy as np
import math, pandas
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

sys.path.append(os.path.split(sys.path[0])[0])

from common import config
from common import datasets
from common import misc
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

    f = open(config.AVA_QUALITY_LABELS_FILE, "r")
    for i, line in enumerate(f):
        if limit_num_pictures:
            if i >= limit_num_pictures:
                logging.info('Reached the developer specified line limit')
                break
        line_array = line.split()
        picture_name = line_array[1]
        aesthetic_values = (line_array[2:])[:10]
        for i in range(0, len(aesthetic_values)): 
            aesthetic_values[i] = int(aesthetic_values[i])
        # pic_label_dict[picture_name] = np.asarray(aesthetic_values).argmax()
        pic_label_dict[picture_name] = np.mean(np.asarray(aesthetic_values))
    # logging.info('label dictionary completed')
    return pic_label_dict


def get_ava_content_labels(limit_num_pictures):
    """ Similar to get_ava_labels but for classification purposes
    
    :rtype: (dict: path->label) dictionary mapping string path to a specific 
        image to int classification label                
    """
    pic_label_dict = {}
    try:
        f = open(config.AVA_CONTENT_LABELS_FILE, "r")
        for i, line in enumerate(f):
            if limit_num_pictures:
                if i >= limit_num_pictures:
                    # logging.info('Reached the developer specified line limit')
                    break
            line_array = line.split()
            picture_name = line_array[1]
            classifications = (line_array[12:])[:-1]
            for i in range(0, len(classifications)): 
                classifications[i] = int(classifications[i])
            pic_label_dict[picture_name] = classifications
        # logging.info('label dictionary completed')
        return pic_label_dict
    except OSError:
        logging.error('Unable to open label file, exiting program')
        sys.exit(1)