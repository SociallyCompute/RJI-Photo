from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import math, pandas
import matplotlib.image as mpimg
from PIL import Image, ImageFile
import logging
import os.path
from os import path
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.split(sys.path[0])[0])

from common import config

def build_image_matrix(image_path):
    image_list = []
    for root, _, files in os.walk(image_path, topdown=True):
        for name in files:
            if name.endswith('.JPG', '.PNG'):
                image_list.append(np.array(Image.open(str(os.path.join(root, name)))).flatten())
    image_matrix = np.hstack(image_list) #done to comply with n_samples x n_features of DBSCAN
    return image_matrix

def cluster_dbscan(image_matrix, eps, minPts):
    #need to ensure image_matrix is (n_samples, n_features)
    db = DBSCAN(eps=eps, min_samples=minPts).fit(image_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(image_matrix, labels))

im_mat = build_image_matrix(config.MISSOURIAN_IMAGE_PATH)
