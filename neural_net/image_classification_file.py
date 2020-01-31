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
        # print(tuple_with_path)
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
    labels_file = open("labeled_images.txt", "w")
    none_file = open("unlabeled_images.txt", "w")
    # _transform = transforms.Compose([transforms.ToTensor()])

    # data = ImageFolderWithPaths(data_dir, transform=_transform)
    # data = ImageFolderWithPaths(data_dir, transform=transforms.Compose([transforms.ToTensor()]))

    # data_loader = torch.utils.data.DataLoader(data)#, num_workers=4)
    data_loader = torch.utils.data.DataLoader(ImageFolderWithPaths(data_dir, transform=transforms.Compose([transforms.ToTensor()])))

    try:
        for i, data in enumerate(data_loader):
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
                    ratings.append(xmp_string[26] + 2)
                    labels_file.write(xmp_string[26] + ", " + str(path) + ", " + str(i))
                else:
                    ratings.append(0)
                    bad_indices.append(i)
                    none_file.write(xmp_string[26] + ", " + str(path) + ", " + str(i))
    except Exception as e:
        print("There was an error on image #{}: {}".format(i, e))
    labels_file.close()
    none_file.close()
# print(counter)

# Define our data transforms to get all our images the same size
def build_dataloaders():
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
    print("Number of pictures in subdirectories: {}".format(num_pictures))

    # Shuffle pictures and split training set
    # indices = list(range(num_pictures))
    # print("Head of indices: {}".format(indices[:10]))

    split = int(np.floor(valid_size * num_pictures))
    print("Split index: {}".format(split))

    # may be unnecessary with the choice of sampler below
    #     np.random.shuffle(indices)
    #     print("Head of shuffled indices: {}".format(indices[:10]))

    train_idx, test_idx = rated_indices, bad_indices#indices[split:], indices[:split]
    print("Size of training set: {}, size of test set: {}".format(len(train_idx), len(test_idx)))

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
    print("Device that will be used: {}".format(device))

    vgg16.to(device) # loads the model onto the device (CPU or GPU)
    return train_loader, test_loader

def change_fully_connected_layer():
    # change the number of classes 
    vgg16.classifier[6].out_features = 10

    for param in vgg16.parameters():
        param.requires_grad = False #freeze all convolution weights
    network = list(vgg16.classifier.children())[:-1] #remove fully connected layer
    network.extend([nn.Linear(4096, 10)]) #add new layer of 4096->100 (rating scale with 1 decimal - similar to 1 hot encoding)
    vgg16.classifier = nn.Sequential(*network)
    vgg16

def train_data_function(train_loader):
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9) # optimizer
    vgg16 #print out the model to ensure our network is correct

    vgg16.train() # set model to training model
    num_epochs = 2 
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        for i, data in enumerate(train_loader,0):
            if limit_num_pictures:
                if i > limit_num_pictures:
                    break
            inputs, _, _, label = data
            label = torch.LongTensor([int(label[0])])
            # print(label)
            optimizer.zero_grad()
            output = vgg16(inputs)
            loss = criterion(output, torch.LongTensor([int(label[0])]))
            running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            num_correct += (preds == label).sum().item()
            loss.backward()
            optimizer.step()
        
            if i % 2000 == 1999:
                # print(running_loss/2000)
                running_loss = 0
            # print("Completed training output for image #{}: {}".format(i, output))
        # training_loss = running_loss/len(train_loader.dataset)
        # training_accuracy = 100 * num_correct/len(train_loader.dataset)
        # print("Training accuracy: {}, Training loss: {}".format(training_accuracy, training_loss))
    torch.save(vgg16.state_dict(), 'models/Jan29_All_2017_Fall_Dump_only_labels_10scale.pt')

def test_data_function(test_loader):
    limit_num_pictures = 5
    vgg16.eval()
    # testing_loss = 0
    # testing_accuracy = 0
    # running_loss = 0.0
    # num_correct = 0
    for i, data in enumerate(test_loader):
        if limit_num_pictures:
            if i > limit_num_pictures:
                break
        # inputs, _, path, label = data
        inputs, _, _, _ = data
        # print(label)
        output = vgg16(inputs)
        # loss = criterion(output, label)

        # running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        print(preds)
        # num_correct += (preds == label).sum().item()
        print("Classification for test image #{}: {}".format(i, output))

    # testing_loss = running_loss/len(test_loader.dataset)
    # testing_accuracy = 100. * num_correct/len(test_loader.dataset)


"""
SCRIPT GLOBAL VARS
"""
# root directory where the images are stored
data_dir = "/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump"#/Fall"#/Dump"
# ratings = None
limit_num_pictures = False #limit_num_pictures = 2000
rated_indices = []
ratings = []
bad_indices = []
# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)


"""
SCRIPT EXECUTION
"""
def run():
    get_color_class_from_xmp()
    train, test = build_dataloaders()
    change_fully_connected_layer()
    train_data_function(train)
    test_data_function(test)

