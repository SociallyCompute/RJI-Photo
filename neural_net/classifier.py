'''
SCRIPT IMPORTS
'''
#standard ML/Image Processing imports
import numpy as np
import pandas as pd
import math, pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)

# root directory where the images are stored
data_dir = "/mnt/md0/reynolds/ava-dataset/"
label_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"
tags_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/tags.txt"
limit_lines = 1000000

tag_mapping = {}
f = open(tags_file, "r")
for i, line in enumerate(f):
    if i >= limit_lines:
        break
    line_array = line.split()
    tag_mapping[int(line_array[0])] = line_array[1]
tag_mapping[0] = "Miscellaneous"
# print(tag_mapping)

pic_label_dict = {}
f = open(label_file, "r")
for i, line in enumerate(f):
    if i >= limit_lines:
        break
    line_array = line.split()
#     print(line_array)
    picture_name = line_array[1]
    # print(picture_name)
    classifications = (line_array[12:])[:-1]
#     print(classifications)
    for i in range(0, len(classifications)): 
#         classifications[i] = tag_mapping[int(classifications[i])]
        classifications[i] = int(classifications[i])
    # print(max(classifications))
    pic_label_dict[picture_name] = classifications
# print(pic_label_dict)

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
#         print(tuple_with_path)
        return tuple_with_path


_transform = transforms.Compose([transforms.ToTensor()])

data = ImageFolderWithPaths(data_dir, transform=_transform)

data_loader = torch.utils.data.DataLoader(data)#, num_workers=4)

limit_num_pictures = 1000000

# Define our data transforms to get all our images the same size
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_size = 0.2 # percentage of data to use for test set

test_data = data
train_data = data
num_pictures = len(train_data)

# Shuffle pictures and split training set
indices = list(range(num_pictures))

split = int(np.floor(valid_size * num_pictures))
train_idx, test_idx = indices[split:], indices[:split]#rated_indices, bad_indices
#print("Size of training set: {}, size of test set: {}".format(len(train_idx), len(test_idx)))

# Define samplers that sample elements randomly without replacement
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

# Define data loaders, which allow batching and shuffling the data
train_loader = torch.utils.data.DataLoader(train_data,
               sampler=train_sampler, batch_size=1)#, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data,
               sampler=test_sampler, batch_size=1)#, num_workers=4)

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# we load the pretrained model, the argument pretrained=True implies to load the ImageNet 
#     weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)
vgg16.to(device) # loads the model onto the device (CPU or GPU)

for param in vgg16.parameters():
    param.requires_grad = False #freeze all convolution weights
network = list(vgg16.classifier.children())[:-1] #remove fully connected layer
network.extend([nn.Linear(4096, 67)]) #add new layer of 4096->100 (rating scale with 1 decimal - similar to 1 hot encoding)
vgg16.classifier = nn.Sequential(*network)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9) # optimizer

vgg16.train() # set model to training model
num_epochs = 3 
training_loss = 0
training_accuracy = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    num_correct = 0
    for i, data in enumerate(train_loader,0):
        #print(i)
        for j in range(2): #done so we can utilize both labels
            if limit_num_pictures:
                if i > limit_num_pictures:
                    break
            inputs, _, path = data
            path = path[0]
            path_array = path.split('/')
            pic_name = path_array[-1]
    #         print(pic_name)
    #         print(pic_label_dict[pic_name.split('.')[0]])
    #         label = torch.LongTensor(pic_label_dict[pic_name.split('.')[0]])
            label = pic_label_dict[pic_name.split('.')[0]][j]
#             print(tag_mapping[label])
            label = torch.LongTensor([label])
#             print('inputs shape is: {}'.format(inputs.shape))
#             print('label shape is: {}'.format(label.shape))
#             print('label is : {}'.format(label))
            optimizer.zero_grad()
            output = vgg16(inputs)
    #         print('output shape is: {}'.format(output.shape))
    #         print(output, label)
            loss = criterion(output, label)
            running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            num_correct += (preds == pic_label_dict[pic_name.split('.')[0]][j]).sum().item()
            loss.backward()
            optimizer.step()
    training_loss = running_loss/len(train_loader.dataset)
    training_accuracy = 100 * num_correct/len(train_loader.dataset)

torch.save(vgg16.state_dict(), 'models/CLASSIFICATION_Feb3_All_AVA_only_training.pt')