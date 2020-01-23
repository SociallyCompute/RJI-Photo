"""

SCRIPT IMPORTS

"""
#standard ML/Image Processing imports
import numpy as np
import math, pandas
import matplotlib.image as mpimg
from PIL import Image

#pytorch imports
import torch
import torch.optim as optim
import torchvision.models as models

from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# no one likes irrelevant warnings
import warnings  
warnings.filterwarnings('ignore')


"""

SCRIPT CLASSES

"""
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
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        #print(tuple_with_path)
        return tuple_with_path

class ImageFolderWithPathsAndRatings(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPathsAndRatings, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path
        # set rating
        try:
            tuple_with_path_and_rating = (tuple_with_path + (ratings[index],))
        except:
            tuple_with_path_and_rating = (tuple_with_path + (torch.FloatTensor([0]),))
        return tuple_with_path_and_rating


"""

SCRIPT FUNCTIONS

"""
def find_size_bounds(limit_num_pictures=None):
    """ Will print and return min/max width/height of pictures in the dataset 
    :param limit_num_pictures - limits the number of pictures analyzed if you purposefully 
        want to work with a smaller dataset
    """
    data = ImageFolderWithPaths(data_dir)
    print(data[0][0].size)
    max_h = (data[0][0]).size[1]
    min_h = data[0][0].size[1]
    max_w = data[0][0].size[0]
    min_w = data[0][0].size[0]
    try:
        for (i, pic) in enumerate(data):
            #if we are limiting pics
            if limit_num_pictures:
                if i > limit_num_pictures:
                    break
            print(pic[0].size) # print all size dimensions
            
            # check width records
            if pic[0].size[0] > max_w:
                max_w = pic[0].size[0]
            elif pic[0].size[1] < min_w:
                min_w = pic[0].size[0]

            # check height records
            if pic[0].size[1] > max_h:
                max_h = pic[0].size[1]
            elif pic[0].size[1] < min_h:
                min_h = pic[0].size[1]
    except Exception as e:
        print(e)
        print("error occurred on pic {} number {}".format(pic, i))

    print("Max/min width: {} {}".format(max_w, min_w))
    print("Max/min height: {} {}".format(max_h, min_h))
    return min_w, max_w, min_h, max_h
#find_size_bounds(limit_num_pictures=1000)

# load data and apply the transforms on contained pictures
def get_color_class_from_xmp():
    labels_file = open("image_splits/full_labeled_images.txt", "w")
    none_file = open("image_splits/full_unlabeled_images.txt", "w")
    # _transform = transforms.Compose([transforms.ToTensor()])

    # data = ImageFolderWithPaths(data_dir, transform=_transform)
    # data = ImageFolderWithPaths(data_dir, transform=transforms.Compose([transforms.ToTensor()]))

    # data_loader = torch.utils.data.DataLoader(data)#, num_workers=4)
    data_loader = torch.utils.data.DataLoader(ImageFolderWithPaths(data_dir, transform=transforms.Compose([transforms.ToTensor()])))

    
    for i, data in enumerate(data_loader):
        try:
            if limit_num_pictures:
                if i > limit_num_pictures:
                    break
            # inputs, labels, path = data
            _, _, path = data
            path = path[0].rstrip()
            with open(path, "rb") as f:
                img = f.read()
            img_string = str(img)
            xmp_start = img_string.find('photomechanic:ColorClass')
            xmp_end = img_string.find('photomechanic:Tagged')
            if xmp_start != xmp_end:
                xmp_string = img_string[xmp_start:xmp_end]
                if xmp_string[26] != "0":
                    print(xmp_string[26] + " " + str(path) + "\n\n")
                    rated_indices.append(i)
                    ratings.append(xmp_string[26])
                    labels_file.write(xmp_string[26] + ", " + str(path) + ", " + str(i))
                else:
                    ratings.append(0)
                    bad_indices.append(i)
                    none_file.write(xmp_string[26] + ", " + str(path) + ", " + str(i))
        except Exception as e:
            print("There was an error on image #{}: {}".format(i, e))
    labels_file.close()
    none_file.close()

"""

SCRIPT GLOBAL VARS

"""
# root directory where the images are stored
data_dir = "/mnt/md0/mysql-dump-economists/Archives"#/2017/Fall/Dump"#/Fall"#/Dump"
# ratings = None
limit_num_pictures = False #limit_num_pictures = 2000
rated_indices = []
ratings = []
bad_indices = []
# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)

get_color_class_from_xmp()
