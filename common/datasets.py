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
import os
warnings.filterwarnings('ignore')

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
AdjustedDataset
    Input: DatasetFolder object
    Attributes: 
        transform: (Tensor Object) Description of how to transform an image.
        classes: 
        class_to_idx: (dict) pairs of (image, class).
        samples: (list) List of (sample_path, class_index) tuples.
        targets: (list) class_index value for each image in dataset.
    Methods:
        __init__
        __getitem__
        __len__
        _find_classes
        get_class_dict
        make_dataset
        pil_loader
        
    Comments:
        Based on this - https://discuss.pytorch.org/t/custom-label-for-torchvision-imagefolder-class/52300/8
        Adjusts dataset to prep for dataloader, map labels to images
"""
class AdjustedDataset(datasets.DatasetFolder):
    """
    __init__
        Input:
            image_path: (string) Folder where the image samples are kept.
            class_dict: (dict) pairs of (picture, rating)
            transform: (Object) Image processing transformations.
    """
    def __init__(self, image_path, class_dict, dataset, classification_subject, device, transform=None):
        self.target_transform = None
        self.classification_subject = classification_subject
        self.device = device
        self.transform = transform
        self.dataset = dataset
        
        # (list) List of the class names.
        self.classes = [i+1 for i in range(67 if self.classification_subject == 'content' else 10)]
        self.class_to_idx = {i+1 : i for i in range(67 if self.classification_subject == 'content' else 10)}
        self.samples = self.make_dataset(image_path, class_dict)
        self.targets = [s[1] for s in self.samples]

    """
    __getitem__
        Input:
            index: (int) Item index
        Output:
            tuple: (tensor, int) where target is class_index of target_class. Same as DatasetFolder object
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.pil_loader(path) #transform Image into Tensor
        if self.transform is not None:
            sample = self.transform(sample).to(self.device)
        if self.target_transform is not None:
            target = self.target_transform(target).to(self.device)
        return sample, target

    
    def make_dataset(self, root, class_to_idx):
        """ .... description ....
        
        :param root: (string) root path to images
        :param class_to_idx: (dict: string, int) image name, mapped to class
        :rtype: [(path, class)] list of paths and mapped classes
        """
        
        images = []
        root_path = os.path.expanduser(root)
        for r, _, fnames in os.walk(root_path):
            for fname in sorted(fnames):
                path = os.path.join(r, fname)

                # AVA dataset has lowercase, Missourian usable pictures 
                #     are uppercase, unusable are lowercase
                if ((path.lower().endswith(('.png', '.jpg')) \
                    and self.dataset == 'ava') or (path.endswith('.JPG'))):
                    
                    item = (path, class_to_idx[fname.split('.')[0]])
                    images.append(item)

        return images

    """
    pil_loader
        Input:
            full_path: (string) path to image for loading
        Output:
            RGB PIL image type
    """
    def pil_loader(self, full_path):
        image = Image.open(full_path)
        image = image.convert('RGB')
        return image
