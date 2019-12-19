#standard ML/Image Processing imports
import numpy as np
import math, pandas
import matplotlib.image as mpimg

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


home = "../../../../../mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump/Cherryhomes, Ellie"
vgg16 = models.vgg16(pretrained=True)
path_list = []

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])

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
        return tuple_with_path

def load_split_train_test(datadir, valid_size = .2):
    
    # Helper/controller params for checking size
    find_size_bounds = False #set to true if you are looking for min/max dims in the current set, false if you want them to be resized
    limit_num_pictures = 500 #set to null if you want no limit
    
    # load data and apply the transforms on contained pictures
    train_data = ImageFolderWithPaths(datadir, transform=_transform)
    test_data = ImageFolderWithPaths(datadir, transform=_transform)   
    
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

def retrieve_xmp_tags():
    for path in path_list:
#         path = path.rstrip()
        with open(path, "rb") as f:
            img = f.read()
        img_string = str(img)
        xmp_start = img_string.find('<x:xmpmeta')
        xmp_end = img_string.find('</x:xmpmeta')
        if xmp_start != xmp_end:
            xmp_string = img_string[xmp_start:xmp_end+12]
            print(xmp_string + '\n\n\n')

# def run_k_means_files():
#     count = 0
#     limit = 150
#     print("Percent done: {}%".format(count/limit*100))
#     for inputs, labels, paths in trainloader:
#         print('\n{}'.format(paths))
#         print(inputs.size())
#         mat4d = inputs
#         mat4d = mat4d[0::2,0::2,:,:]
#         mat2d = mat4d.resize_((mat4d.shape[1] * mat4d.shape[2]), mat4d.shape[3])
        
#         add_to_list_file(paths[0], mat2d)
#         count = count + 1
#         print("Percent done: {}%".format(count/limit*100))
#         if(count >= limit):
#             break

def change_fc():
    for param in vgg16.parameters():
        param.requires_grad = False #freeze all weights
    network = list(vgg16.classifier.children())[:-1] #remove fully connected layer
    network.extend([nn.Linear(4096, 100)]) #add new layer of 4096->100 (rating scale with 1 decimal - similar to 1 hot encoding)
    vgg16.classifier = nn.Sequential(*network)
    print(vgg16) #print out the model to ensure our network is correct

def train_network(max_epoch, learning_rate, mom, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=mom)
    for epoch in range(max_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            print(data)
            inputs, labels = data
            optimizer.zero_grad()
            
            output = vgg16(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(running_loss/2000)
                running_loss = 0
            print("Finished")

def run_vgg(training, testing):
    # training, testing = load_split_train_test(home, .2)
    # for _,_,paths in training:
    #     path_list.append(paths)
    # change_fc()
    train_network(2, 0.4, 0.9, training)
    vgg16.eval()
    for _,data in enumerate(testing,0):
        output = vgg16(data[0])
        max_val = torch.argmax(output)
        print('labels: ' + str(data[1]))
        print(data[2])
        print(max_val)
        print("\n\n\n")
        # loss = criterion(output, labels)
        # print(output)


# if __name__ == "__main__":
#     run_vgg()
# run_k_means_files()