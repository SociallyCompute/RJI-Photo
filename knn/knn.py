from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#create a dictionary of pictures
pics = {}
np_pics = []
second_pics = []
reduced_pics = []
pca = PCA(n_components=25)

def run_knn():
    print("np_pics count: " + str(len(np_pics)))
    for i in np_pics:
        i = pca.fit_transform(i) #standardize images to 2000 x 25
        reduced_pics.append(i.flatten()) #flatten so image is a vector        

    print(len(reduced_pics))
    p = np.vstack(reduced_pics)
    print("Shape of p: " + str(p.shape))
    plt.scatter(p[:,0], p[:,1], label='True Position')
    plt.show()
    # km = KMeans(n_clusters=4)
    # km.fit(training_data)
    return

def add_to_list(loc,f):
    im = Image.open(loc + '\\' + f)
    #this should never fire, it would mean there is a duplicate picture
    if(f in pics):
        return
    #add to pics dictionary with file name for key and add to np list
    im = im.convert('1')
    pics[f] = im
    mat = np.array(im)
    if mat.shape[1] == 2000:
        mat = mat.transpose()
        np_pics.append(mat)
    else:
        if mat.shape[0] == 2000:
            np_pics.append(mat)
        else:
            second_pics.append(mat)
    print(f + " : " + str(np_pics[-1].shape))
    return