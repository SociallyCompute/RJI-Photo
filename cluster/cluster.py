from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#create a dictionary of pictures
pics = {}
index_set = {}
np_pics = []
second_pics = []
reduced_pics = []


def pca_compress(n):
    pca = PCA(n_components=n)
    for i in np_pics:
        i = pca.fit_transform(i) #standardize images to 2000 x 25
        reduced_pics.append(i.flatten()) #flatten so image is a vector

def run_kmeans(clusters):
    km = KMeans(n_clusters=clusters)
    km.fit(reduced_pics)

    index_set = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)} #index of pictures in data
    # {i: reduced_pics[np.where(km.labels_ == i)] for i in range(km.n_clusters)} #actual pictures
    files = list(pics)
    for i in range(len(index_set)):
        c = index_set[i]
        for v in c:
            print(files[v])
        print('------------')
    # print(pics.keys()[index_set[1]) #change index to the index you want to see which cluster it is in

def run_knn(clusters):
    p = np.vstack(reduced_pics)
    k = knn(clusters)
    print(p)
    print(p.shape)
    k.fit(p, p.shape) 
    index_set = {i: np.where(k.classes_ == i)[0] for i in range(clusters)}
    files = list(pics)
    for i in range(len(index_set)):
        c = index_set[i]
        for v in c:
            print(files[v])
        print('------------')

def add_to_list(loc,f):
    im = Image.open(loc + '\\' + f)
    #this should never fire, it would mean there is a duplicate picture
    if(f in pics):
        return
    #add to pics dictionary with file name for key and add to np list
    # im = im.convert('1') #convert to grayscale
    mat = np.array(im)
    print(mat.shape)
    if mat.shape[1] == 7360:
        pics[f] = im
        mat = mat.transpose()
        np_pics.append(mat)
    else:
        if mat.shape[0] == 7360:
            pics[f] = im
            np_pics.append(mat)
        else:
            second_pics.append(mat)
    print(f + " : " + str(np_pics[-1].shape))
    return