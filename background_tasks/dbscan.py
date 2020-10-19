from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
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
from sqlalchemy.orm import Session
import sqlalchemy as sqla
import pandas

from torch import squeeze

from torchvision import datasets, transforms

sys.path.append(os.path.split(sys.path[0])[0])

from common import config
from common import misc
from common import connections
from common import image_processing

def build_image_matrix(image_path):
    _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])
    image_list = []
    name_list = []
    pca = PCA(n_components='mle')
    for root, _, files in os.walk(image_path, topdown=True):
        for name in files:
            path = os.path.join(root, name)
            logging.info(name)
            if name.endswith('.JPG'):
                #saving image as greyscale so we can save as 2 dimensions rather than 3 for RGB
                pil_img = Image.open(str(os.path.join(root,name))).convert('LA')
                tra = squeeze(_transform(pil_img)) #get rid of dimensions with 1 -> resizing to 224 x 224 instead of 1 x 224 x 224
                image_list.append(image_processing.canny_edge_detection(np.array(tra)).flatten())
                name_list.append(name)
    image_matrix = np.vstack(image_list) #done to comply with n_samples x n_features of DBSCAN
    pca_reduced = pca.fit_transform(image_matrix)
    # logging.info(image_matrix)
    print(pca.n_components_)
    return pca_reduced, name_list

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
    # print("Silhouette Coefficient: %0.3f"
    #     % metrics.silhouette_score(image_matrix, labels))
    
    return labels, n_clusters_

logging.basicConfig(filename='logs/dbscan.log', filemode='w', level=logging.DEBUG)
#how close each picture is to each other (lower is closer, higher is farther)
epsilon = 20
#total number of similar pictures to consider the picture a "core point" in DBSCAN
minimum_points = 2

im_mat, nm_list = build_image_matrix(config.MISSOURIAN_IMAGE_PATH)
labels, clusters = cluster_dbscan(im_mat, epsilon, minimum_points)

session_db_tuple = {}
session_db_tuple['distance_between_points'] = epsilon
session_db_tuple['minimum_points'] = minimum_points
session_db_tuple['num_clusters'] = clusters

session_db, session_table = connections.make_db_connection('cluster_sessions')
result = session_db.execute(session_table.insert().values(session_db_tuple))

session_db, session_table = connections.make_db_connection('cluster_sessions')

data_SQL = sqla.sql.text("""
        SELECT cluster_session_id
        FROM cluster_sessions
        ORDER BY data_collection_time DESC
        LIMIT 1;
        """)

xmp_data = pandas.read_sql(data_SQL, session_db, params={})
logging.info(xmp_data)
xmp_data = xmp_data.set_index('cluster_session_id')
# val = xmp_data['cluster_session_id']
indices = xmp_data.index
index_list = list(indices)
logging.info(index_list[-1])
c_s_id = index_list[-1]

results_db, results_table = connections.make_db_connection('cluster_results')
conn = results_db.connect()
labels_count = len(labels)
nm_list_count = len(nm_list)
print("labels_count: {}\nnm_list_count: {}".format(labels_count, nm_list_count))
print(labels)
print("============")
conn.execute(
    results_table.insert(),
    [
        dict(
            cluster_session_id=int(c_s_id), #.astype(int), #.item() if isinstance(labels[i], np.int64) else c_s_id,
            photo_path=nm_list[i],
            cluster_number=int(labels[i]), #.astype(int), #.item() if isinstance(labels[i], np.int64) else labels[i],
        )
        for i in range(labels_count)
    ],
)


