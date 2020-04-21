from sklearn.cluster import DBSCAN
import numpy as np
import math, pandas
import matplotlib.image as mpimg
from PIL import Image, ImageFile
import logging
import os.path
from os import path
ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_image_matrix(image_path):
    image_list = []
    for root, _, files in os.walk(image_path, topdown=True):
        for name in files:
            if name.endswith('.JPG', '.PNG'):
                image_list.append(np.array(Image.open(str(os.path.join(root, name)))).flatten())
    image_matrix = np.vstack(image_list)
    return image_matrix

def cluster_dbscan(image_matrix, eps, minPts):
    DBSCAN(eps=eps, min_samples=minPts).fit(image_matrix)