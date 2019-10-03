from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#create a dictionary of pictures
pics = {}
index_set = {}
np_pics = []
second_pics = []
reduced_pics = []
color = [0, 0, 0]
colors = []
count = 0
pca = PCA(n_components=25)

def run_knn():
    print("np_pics count: " + str(len(np_pics)))
    for i in np_pics:
        i = pca.fit_transform(i) #standardize images to 2000 x 25
        reduced_pics.append(i.flatten()) #flatten so image is a vector
        #this is disgusting, but it is meant as a quick fix for coloring
        # if(color[0] > 255):
        #     color[0] = 0
        #     color[1] = color[1] + 15
        #     if(color[1] > 255):
        #         color[1] = 0
        #         color[2] = color[2] + 15
        #         if(color[2] > 255):
        #             color[2] = 0
        # colors.append((color[0], color[1], color[2])) 


    print(len(reduced_pics))
    p = np.vstack(reduced_pics)
    print("Shape of p: " + str(p.shape))
    # for i in colors:
    #     count = count + 1
    plt.scatter(p[:,0], p[:,1], label='True Position')
    plt.show()
    km = KMeans(n_clusters=15)
    km.fit(reduced_pics)

    index_set = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)} #index of pictures in data
    # {i: reduced_pics[np.where(km.labels_ == i)] for i in range(km.n_clusters)} #actual pictures
    print(index_set)
    # print(pics.keys()[index]) #change index to the index you want to see which cluster it is in
    return

def add_to_list(loc,f):
    im = Image.open(loc + '\\' + f)
    #this should never fire, it would mean there is a duplicate picture
    if(f in pics):
        return
    #add to pics dictionary with file name for key and add to np list
    im = im.convert('1')
    mat = np.array(im)
    if mat.shape[1] == 2000:
        pics[f] = im
        mat = mat.transpose()
        np_pics.append(mat)
    else:
        if mat.shape[0] == 2000:
            pics[f] = im
            np_pics.append(mat)
        else:
            second_pics.append(mat)
    print(f + " : " + str(np_pics[-1].shape))
    return