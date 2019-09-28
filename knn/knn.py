from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#create a dictionary of pictures
pics = {}
np_pics = []
reduced_pics = []
pca = PCA(n_components=2)

def run_knn():
    for i in np_pics:
        if(i.ndim > 2):
            nsamples, x, y = i.shape
            i = i.reshape((nsamples,x*y))
        #currently here, struggling with condensing to 1 sample
        i = i.flatten()
        i = i.reshape(1, -1)
        #seems to think min(samples, features) = 1 although dims is much higher
        reduced_pics.append(pca.fit_transform(i))        

    print(reduced_pics)
    print(len(reduced_pics))
    print(reduced_pics[0].shape)
    for i in reduced_pics:
        plt.scatter(reduced_pics[i][:,0], reduced_pics[i][:,1], label='True Position')
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
    pics[f] = im
    np_pics.append(np.array(im))
    return