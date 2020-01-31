#standard ML/Image Processing imports
import numpy as np
import pandas as pd
import math, pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# root directory where the images are stored
data_dir = "/mnt/md0/reynolds/ava-dataset/"
label_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"

ratings = None
pic_label_dict = {}
limit_lines = 1000000
f = open(label_file, "r")
for i, line in enumerate(f):
    if i >= limit_lines:
        break
    line_array = line.split()
#     print(line_array)
    picture_name = line_array[1]
    # print(picture_name)
    temp = line_array[2:]
    # print(temp)
    aesthetic_values = temp[:10]
    # print(aesthetic_values)
    for i in range(0, len(aesthetic_values)): 
        aesthetic_values[i] = int(aesthetic_values[i])
    # print(max(aesthetic_values))
    pic_label_dict[picture_name] = np.asarray(aesthetic_values).argmax()
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
            rating = ratings[index] - 1 if ratings[index] > 0 else ratings[index]
            tuple_with_path_and_rating = (tuple_with_path + (rating,))
        except:
            tuple_with_path_and_rating = (tuple_with_path + (torch.FloatTensor([0]),))
        return tuple_with_path_and_rating

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

    #print("Max/min width: {} {}".format(max_w, min_w))
    #print("Max/min height: {} {}".format(max_h, min_h))
    return min_w, max_w, min_h, max_h
#find_size_bounds(limit_num_pictures=1000)
# load data and apply the transforms on contained pictures

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

# load data and apply the transforms on contained pictures
train_data = ImageFolderWithPathsAndRatings(data_dir, transform=_transform)
test_data = ImageFolderWithPathsAndRatings(data_dir, transform=_transform)   

num_pictures = len(train_data)
#print("Number of pictures in subdirectories: {}".format(num_pictures))

# Shuffle pictures and split training set
indices = list(range(num_pictures))
# print("Head of indices: {}".format(indices[:10]))

split = int(np.floor(valid_size * num_pictures))
# print("Split index: {}".format(split))

# may be unnecessary with the choice of sampler below
#     np.random.shuffle(indices)
#     print("Head of shuffled indices: {}".format(indices[:10]))

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
#print("Device that will be used: {}".format(device))

# we load the pretrained model, the argument pretrained=True implies to load the ImageNet 
#     weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)
vgg16.to(device) # loads the model onto the device (CPU or GPU)

for param in vgg16.parameters():
    param.requires_grad = False #freeze all convolution weights
network = list(vgg16.classifier.children())[:-1] #remove fully connected layer
network.extend([nn.Linear(4096, 10)]) #add new layer of 4096->100 (rating scale with 1 decimal - similar to 1 hot encoding)
vgg16.classifier = nn.Sequential(*network)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9) # optimizer

# vgg16 #print out the model to ensure our network is correct

vgg16.train() # set model to training model
num_epochs = 1 
training_loss = 0
training_accuracy = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    num_correct = 0
    for i, data in enumerate(train_loader,0):
        #print(i)
        if limit_num_pictures:
            if i > limit_num_pictures:
                break
        inputs, _, path, label = data
        path = path[0]
        path_array = path.split('/')
        pic_name = path_array[-1]
#         print(pic_name)
#         print(pic_label_dict[pic_name.split('.')[0]])
#         label = torch.LongTensor(pic_label_dict[pic_name.split('.')[0]])
        label = pic_label_dict[pic_name.split('.')[0]]
        label = torch.LongTensor([label])
#         print('inputs shape is: {}'.format(inputs.shape))
#         print('label shape is: {}'.format(label.shape))
        #print('label is : {}'.format(label))
        optimizer.zero_grad()
        output = vgg16(inputs)
#         print('output shape is: {}'.format(output.shape))
#         print(output, label)
        loss = criterion(output, label)
        running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        num_correct += (preds == pic_label_dict[pic_name.split('.')[0]]).sum().item()
        loss.backward()
        optimizer.step()
    

        #print("Completed training output for image #{}: {}".format(i, output))
        if epoch == 0 and i % 20 == 0:
            #print('test')
#             fig = plt.figure(figsize=(16, 4))
            columns = 3
            rows = 1
            short_name = ''
#             short_name.join(path[0].split('/')[8:])
#             print(short_name)
#             img = mpimg.imread(path[0])
#             fig.add_subplot(rows, columns, 1)
#             plt.imshow(img)
#             plt.xticks([])
#             plt.yticks([])
#             plt.show()
    training_loss = running_loss/len(train_loader.dataset)
    training_accuracy = 100 * num_correct/len(train_loader.dataset)
    #print("Training accuracy: {}, Training loss: {}".format(training_accuracy, training_loss))
torch.save(vgg16.state_dict(), 'models/Jan29_All_AVA_only_training.pt')
