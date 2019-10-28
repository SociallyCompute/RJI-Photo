import sys
import os
import cluster
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.decomposition import PCA


pics = {}
index_set = {}
np_pics = []
second_pics = []
reduced_pics = []
dump_files = []
edited_files = []
labels = {}

# def print_menu():
#     print("Which algorithm would you like to train?")
#     print("1: K-Means")
#     print("2: KNN")
#     print("3: Convolusional Neural Network")
#     alg = input("Your choice: ")
#     print("What dataset would you like to use?")
#     print("1: root")
#     print("2: Custom path")
#     data = input("Your choice: ")
#     print("What folder do you want to stop on?")
#     stop = input("Your choice: ")
#     return alg, data, stop

# def run(root, halt, alg):
#     # count = 0
#     fi = open('labels.txt', 'a')
#     for(loc,dirs,files) in os.walk('/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/',topdown=True):
#         for f in files:
#             if(f.lower().endswith('.jpg')):
#                 dump_files.append(f.lower())
    
#     for(loc,dir,files) in os.walk('/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Edited/',topdown=True):
#         for f in files:
#             if f in dump_files:
#                 labels[f] = 1
#                 fi.write(str(f) + " 1\n")
#             else:
#                 labels[f] = 0
#                 fi.write(str(f) + " 0\n")

#     print(labels.values())


def load_split_train_test(datadir, valid_size = .2):
    
    # Helper/controller params for checking size
    find_size_bounds = False #set to true if you are looking for min/max dims in the current set, false if you want them to be resized
    limit_num_pictures = 1000 #set to null if you want no limit
    
    # define transforms to resize (look into 'Rescale'??) images to desired size and transform them into tensors
    """ more on other transforms (RandomHorizontalFlip ("double" dataset") and RandomResizedCrop (increase 
        robustness and also artificially "increases" our dataset) from a SO post: 
        https://stackoverflow.com/questions/50963295/pytorch-purpose-of-images-preprocessing-in-the-transfer-learning-tutorial
    """
    train_transforms = transforms.Compose([transforms.Resize((425,242)),transforms.ToTensor(),]) if not find_size_bounds else transforms.Compose([transforms.ToTensor(),])
    test_transforms = transforms.Compose([transforms.Resize((425,242)),transforms.ToTensor(),]) if not find_size_bounds else transforms.Compose([transforms.ToTensor(),])
    
    # load data and apply the transforms on contained pictures
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)   
    
    maxh = 0
    minh = 10000
    maxw = 0
    minw = 10000
    if find_size_bounds:
        try:
            for (i, pic) in enumerate(train_data):
                #if we are limiting pics
                if limit_num_pictures:
                    if i > limit_num_pictures:
                        break
                print(pic[0].size())
                if pic[0].size()[1] > maxw:
                    maxw = pic[0].size()[1]
                elif pic[0].size()[1] < minw:
                    minw = pic[0].size()[1]

                if pic[0].size()[2] > maxh:
                    maxh = pic[0].size()[2]
                elif pic[0].size()[2] < minh:
                    minh = pic[0].size()[2]
        except Exception as e:
            print(e)
            print("error occurred on pic {} number {}".format(pic, i))
    
        print("Max/min width: {} {}".format(maxw, minw))
        print("Max/min height: {} {}".format(maxh, minh))
    
    num_pictures = len(train_data)
    print("Number of pictures in subdirectories: {}".format(num_pictures))
    
    # Shuffle pictures and split training set
    indices = list(range(num_pictures))
    print("Head of indices: {}".format(indices[:10]))
    
    split = int(np.floor(valid_size * num_pictures))
    print("Split index: {}".format(split))
    
    # may be unnecessary with the choice of sampler below
#     np.random.shuffle(indices)
#     print("Head of shuffled indices: {}".format(indices[:10]))
    
    train_idx, test_idx = indices[split:], indices[:split]
    print("Size of training set: {}, size of test set: {}".format(len(train_idx), len(test_idx)))
    
    # Define samplers that sample elements randomly without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    # Define data loaders, which allow batching the data, shuffling the data, and 
    #     loading the data in parallel using multiprocessing workers
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=1, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=1, num_workers=4)
    return trainloader, testloader
# rename("/mnt/md0/mysql-dump-economists/Archives/1999/Fall/Dump/Beard, Rebecca")

pics = {}
np_pics = []
def add_to_list_file(line, matrix):
    pics[line] = matrix
    print(matrix.shape)
    np_pics.append(matrix.flatten())

def run_kmeans_file(matrix, clusters):
    f = open('groupings.txt', 'a')
    km = KMeans(n_clusters=clusters) # establishing the KMeans model
    km.fit(matrix)
    index_set = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)} #index of pictures in data
    files = list(pics)
    for i in range(len(index_set)):
        c = index_set[i]
        for v in c:
            f.write(files[v] + "\n")
            print(files[v])
        f.write('-----------')
        print('------------')
    f.close()

def pca_file(n):
    all_samples = np.vstack(np_pics)
    print(all_samples.shape)
    pca = PCA(n_components=n) # establish the PCA model
    return pca.fit_transform(all_samples)

def run_with_trainloader():    
    """define the train / validation dataset loader, using the SubsetRandomSampler for the split"""

    data_dir = "../../../../../mnt/md0/mysql-dump-economists/Archives"#/Fall"#/Dump"
    trainloader, testloader = load_split_train_test(data_dir, .2)
    count = 0
    for inputs, labels in trainloader:
        count = count + 1
        line = "../../../../.." + inputs.rstrip()
        im = Image.open(line)
        exif_data = im._getexif() # pulling out exif data from image and printing
        print(exif_data)
        mat3d = np.array(im) # convert image to numpy matrix
        mat3d = mat3d[0::2,0::2,:] # cutting image pixels in half
        mat2d = mat3d.reshape((mat3d.shape[0] * mat3d.shape[1]), mat3d.shape[2])
        add_to_list_file(line, mat2d)
        if(count >= 250):
            break
    matrix = pca_file(int(input("How many components would you like to compress: ")))
    run_kmeans_file(matrix, int(input("How many clusters would you like to group: ")))

# def run_files():
#     count = 0
#     f = open("18048files.txt", "r")
#     for line in f:
#         count = count + 1
#         line = "../../../../.." + line.rstrip()
#         print(line)
#         mat3d = np.array(Image.open(line))
#         mat3d = mat3d[0::2,0::2,:]
#         mat2d = mat3d.reshape((mat3d.shape[0] * mat3d.shape[1]), mat3d.shape[2])
#         # print(mat2d.shape)
#         cluster.add_to_list_file(line, mat2d)
#         if(count >= 250):
#             break
#     matrix = cluster.pca_file(int(input("How many components would you like to compress: ")))
#     cluster.run_kmeans_file(matrix, int(input("How many clusters would you like to group: ")))


#split into 5 groups of 4 years apiece?
#keep relevance in the pictures, was there a specific point in the last 20 years cameras improved?
#have argv[1] be the root folder and argv[2] be the first folder we don't want to pull from
if(__name__ == "__main__"):
    run_with_trainloader()
    # alg, data, stop = print_menu()
    #how can I generalize this without requiring people type this out?
    # if(data == "root" or data == '1'):
    #     # root_dir = '/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/'
    #     root_dir = '/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Edited/'
    # else:
    #     root_dir = data
    # if alg == "1":
    # run_files()
    # else:
    # run(root_dir, stop, alg)
