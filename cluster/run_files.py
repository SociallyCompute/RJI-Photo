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
import PIL.ExifTags
from iptcinfo3 import IPTCInfo
import base64


pics = {}
index_set = {}
np_pics = []
second_pics = []
reduced_pics = []
dump_files = []
edited_files = []
labels = {}
pics = {}
np_pics = []


def add_to_list_file(line, matrix):
    pics[line] = matrix
    print(matrix.shape)
    np_pics.append(matrix.flatten())

def run_kmeans_file(matrix, clusters):
    # f = open('groupings.txt', 'a')
    km = KMeans(n_clusters=clusters) # establishing the KMeans model
    km.fit(matrix)
    index_set = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)} #index of pictures in data
    print(index_set)
    # files = list(pics)
    # for i in range(len(index_set)):
        # c = index_set[i]
        # for v in c:
            # f.write(files[v] + "\n")
            # print(files[v])
        # f.write('-----------')
        # print('------------')
    # f.close()

def pca_file(n):
    all_samples = np.vstack(np_pics)
    print(all_samples.shape)
    pca = PCA(n_components=n) # establish the PCA model
    return pca.fit_transform(all_samples)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # if datasets.folder.has_file_allowed_extension(path, (".jpg", ".JPG")):
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

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
    train_data = ImageFolderWithPaths(datadir, transform=train_transforms)
    test_data = ImageFolderWithPaths(datadir, transform=test_transforms)   
    
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
                   sampler=train_sampler, batch_size=1)#, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=1)#, num_workers=4)
    return trainloader, testloader

def run_files():
    path_files = open("paths.txt", "a")
    count = 0
    limit = 150
    print("Percent done: {}%".format(count/limit*100))
    for inputs, labels, paths in trainloader:
        print('\n{}'.format(paths))
        path_files.write(str(paths[0]) + "\n")
        
#         line = "../../../../.." + labels.rstrip()
#         mat3d = np.array(Image.open(line))
#         mat3d = mat3d[0::2,0::2,:]
#         mat2d = mat3d.reshape((mat3d.shape[0] * mat3d.shape[1]), mat3d.shape[2])
        print(inputs.size())
        mat4d = inputs
#         print(mat4d)
        mat4d = mat4d[0::2,0::2,:,:]
#         print(mat4d)
        mat2d = mat4d.resize_((mat4d.shape[1] * mat4d.shape[2]), mat4d.shape[3])
#         print(mat2d)
        
        add_to_list_file(paths[0], mat2d)
        count = count + 1
        print("Percent done: {}%".format(count/limit*100))
        if(count >= limit):
            path_files.close()
            break
    path_files.close()

'''
    Retrieve exif data as a list. Exif data presents metadata created at the time the picture is taken.
    It contains info such as the camera style, size, location (if applicable), etc. 
    *Note* this does not generally contain info editors added in via Photomechanic, Photoshop, or other editing tools
'''
def get_exif_data():
    exif_data = list()
    paths_file = open("paths.txt", "r")
    for line in paths_file:
        if line.rstrip().lower().endswith('.jpg'): 
            im = Image.open(line.rstrip())
            print(line.rstrip())
            if im._getexif() is None:
                continue
            else:
                exif = { 
                        PIL.ExifTags.TAGS[k]:v 
                        for k, v in im._getexif().items() 
                        if k in PIL.ExifTags.TAGS 
                }
                exif_data.append(exif)
                print(str(exif.keys()) + "\n")
                if 'ColorSpace' in exif:
                    print(str(exif['ColorSpace']) + "\n\n\n")
    return exif_data

'''
    Retrieve International Press Telecommunications Council (IPTC) metadata. IPTC metadata is the primary way info is stored 
    via editors. The most focused on element here is the 221 tag, which represents preference in Photomechanic. This
    tag is stored as a base64 number and read in as a bit stream. Other data IPTC contains includes keywords, categories, 
    authors/editors names, all of which only exist if an editor has added them. Most of this data depends on an individual
    adding it to the metadata some time after creation.
'''
def get_iptc_data():
    iptc_data = list()
    paths_file = open("paths.txt", "r")
    for line in paths_file:
        if line.rstrip().lower().endswith('.jpg'):
            info = IPTCInfo(line.rstrip())
            #print(str(dir(info)) + "\n")
            print(line.rstrip())
            # print(str(info._data))
            # print(str(info._data.keys()))
            # print(info._data['nonstandard_221'])
            iptc_data.append(info)
            if info._data['nonstandard_221'] is None:
                continue
            else:
                '''
                    Photomechanic stores editor preference tags in 221. These values are stored in base64
                    which is 4 strings of 6 numbers, and all images have the format 0:0:0:xxxxxx. 
                    Because the last values are the only ones that change we decode the bit stream into 
                    a string and chop the front off. After we re-encode the data back into a bit stream
                    and attempt to decode the base64 into binary. This can then be converted into usable ASCII
                '''
                val = info._data['nonstandard_221'].decode('utf-8')
                print(str(val))
                # pref = "000000000000000000" + val[-6:]
                # pref = pref.encode('utf-8')
                # print(pref)
                # decoded = base64.decodebytes(pref)
                # print(str(decoded) + "\n\n")
            # iptc_data.append(info)
    return iptc_data

#split into 5 groups of 4 years apiece?
#keep relevance in the pictures, was there a specific point in the last 20 years cameras improved?
#have argv[1] be the root folder and argv[2] be the first folder we don't want to pull from
if(__name__ == "__main__"):
    choice = input('1 for training 2 for printing exif example: ')
    if choice == '1':
        # run_with_trainloader()
        data_dir = "../../../../../mnt/md0/mysql-dump-economists/Archives"#/Fall"#/Dump"
        trainloader, testloader = load_split_train_test(data_dir, .2)
        run_files()
        # exif_d = get_exif_data()
        iptc_d = get_iptc_data()
        # print(exif_d)
    else:
        dump1im = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/Cherryhomes, Ellie/20170822_NAACPForum_EC/20170822_NAACPForum_EC_134.JPG"
        edit1im = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Edited/Cherryhomes, Ellie/20170822_NAACPForum_EC_134.JPG"

        dump2im = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/Cherryhomes, Ellie/20170822_NAACPForum_EC/20170822_NAACPForum_EC_164.JPG"
        edit2im = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Edited/Cherryhomes, Ellie/20170822_NAACPForum_EC_164.JPG"

        dump3im = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/Cherryhomes, Ellie/20171208_recycling_ec/20171208_recylingmizzou_ec_008.JPG"

        dump4im = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/Cherryhomes, Ellie/20170905_BattleTennisGirls_ec/20170905_battletennis_ec_005.JPG"

        dump_info1 = IPTCInfo(dump1im)
        print('dump 1: ' + str(dump_info1._data['nonstandard_221']))
        edit_info1 = IPTCInfo(edit1im)
        print('edit 1: ' + str(edit_info1._data['nonstandard_221']))

        dump_info2 = IPTCInfo(dump2im)
        print('dump 2: ' + str(dump_info2._data['nonstandard_221']))
        edit_info2 = IPTCInfo(edit2im)
        print('edit 2: ' + str(edit_info2._data['nonstandard_221']))

        dump_info3 = IPTCInfo(dump3im)
        print('dump 3: ' + str(dump_info3._data['nonstandard_221']))

        dump_info4 = IPTCInfo(dump4im)
        print('dump 4: ' + str(dump_info4._data['nonstandard_221']))

        # img = Image.open(im)
        # exif = img._getexif()
        # print(exif)
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
