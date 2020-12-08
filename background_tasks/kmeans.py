from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from PIL import Image, ImageFile
import logging
import os.path
from os import path
import sys

import numpy as np

from torchvision import datasets, transforms

sys.path.append(os.path.split(sys.path[0])[0])
from common import config
from common import image_processing

def build_image_matrix(image_path):
    _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
            # transforms.ToTensor()
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])
    image_list = []
    name_list = []
    # pca = PCA(n_components=64)
    for root, _, files in os.walk(image_path, topdown=True):
        for name in files:
            path = os.path.join(root, name)
            logging.info(name)
            if name.endswith('.JPG'):
                #saving image as greyscale so we can save as 2 dimensions rather than 3 for RGB
                pil_img = Image.open(str(os.path.join(root,name))).convert('L')
                # tra = squeeze(_transform(pil_img)) #get rid of dimensions with 1 -> resizing to 224 x 224 instead of 1 x 224 x 224
                tra = _transform(pil_img)
                # print(np.array(tra).shape)
                canny_flattened_image = image_processing.canny_edge_detection(np.array(tra)).flatten()
                # print(canny_flattened_image.shape)
                image_list.append(canny_flattened_image)
                name_list.append(name)
    image_matrix = np.vstack(image_list) #done to comply with n_samples x n_features of DBSCAN
    # print(image_matrix.shape)
    # pca_reduced = pca.fit_transform(image_matrix)
    # logging.info(image_matrix)
    # print(pca.n_components_)
    # return pca_reduced, name_list
    return image_matrix, name_list

im_mat, nm_list = build_image_matrix(config.MISSOURIAN_IMAGE_PATH)

sil = []
kmax = 50

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(im_mat)
  labels = kmeans.labels_
  sil.append(silhouette_score(im_mat, labels, metric = 'euclidean'))

print(sil)