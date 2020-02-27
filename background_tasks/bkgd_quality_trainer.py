"""
SCRIPT IMPORTS
"""
import numpy as np
import math, pandas
import matplotlib.image as mpimg
from PIL import Image

import torch
import torch.optim as optim
import torchvision.models as models

from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import logging
import sys
import os.path
import ntpath
from os import path
from pathlib2 import Path

import warnings  
warnings.filterwarnings('ignore')


"""
ImageFolderWithPaths
    Input: ImageFolder object
    Comments:
        add the path to the tuple supplied from the original ImageFolder.
"""
class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

"""
ImageFolderWithPathsAndRatings
    Input: ImageFolder object
    Comments:
        add the path and rating to the tuple supplied from the original ImageFolder.
"""
class ImageFolderWithPathsAndRatings(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPathsAndRatings, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        # set rating
        try:
            tuple_with_path_and_rating = (tuple_with_path + (ratings[index],))
        except:
            tuple_with_path_and_rating = (tuple_with_path + (torch.FloatTensor([0]),))
        return tuple_with_path_and_rating

"""
AdjustedDataset
    Input: DatasetFolder object
    __init__: 
        root to images (string), image to class dict (dict: string, int), transform operator(transform Object), max class (int)
    Comments:
        expects 1 class per image. Needs to be adjusted to incorporate 2+ classes per image
"""
# https://discuss.pytorch.org/t/custom-label-for-torchvision-imagefolder-class/52300/8
class AdjustedDataset(datasets.DatasetFolder):
    def __init__(self, image_path, class_dict, transform=None):
        """Imports dataset from folder structure.

        Input:
            image_path: (string) Folder where the image samples are kept.
            class_dict: (dict) pairs of (picture, rating)
            transform: (Object) Image processing transformations.

        Attributes: 
            classes: (list) List of the class names.
            class_to_idx: (dict) pairs of (image, class).
            samples: (list) List of (sample_path, class_index) tuples.
            targets: (list) class_index value for each image in dataset.
        """
        super(AdjustedDataset, self).__init__(image_path, self.pil_loader, extensions=('.jpg', '.png', '.PNG', '.JPG'),transform=transform)
        self.transform = transform
        # self.classes = [i+1 for i in range(10)] #classes are 1-10
        # self.class_to_idx = class_dict
        self.classes, self.class_to_idx = self._find_classes(class_dict)
        self.samples = self.make_dataset(image_path, class_dict)
        self.targets = [s[1] for s in self.samples]

    def pil_loader(self, full_path):
        image = Image.open(full_path)
        tensor_sample = transforms.ToTensor(image)
        return tensor_sample

    def _find_classes(self, class_dict):
        classes = list(class_dict.values())
        class_to_idx = {classes[i] : i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, root, class_to_idx):
        '''
        Input:
            1. root: (string) root path to images
            2. class_to_idx: (dict: string, int) image name, mapped to class
        
        Output:
            images: [(path, class)] list of paths and mapped classes
        '''
        images = []
        root_path = os.path.expanduser(root)
        # for target in sorted(class_to_idx.keys()):
        #     d = os.path.join(root, target)
        #     if not os.path.isdir(d):
        #         continue
        for r, _, fnames in os.walk(root_path):
            for fname in sorted(fnames):
                path = os.path.join(r, fname)
                logging.info('path is {}'.format(path))
                if path.lower().endswith(('.png', '.jpg')):
                    item = (path, class_to_idx[fname.split('.')[0]])
                    logging.info('appending item {}'.format(item))
                    images.append(item)

        return images


    def get_class_dict(self):
        """Returns a dictionary of classes mapped to indicies."""
        return self.class_to_idx

    def __getitem__(self, index):
        """Returns tuple: (tensor, int) where target is class_index of
        target_class. Same as DatasetFolder object
        
        Args:
            idx: (int) Index.
        """

        path, target = self.samples[index]
        sample = self.pil_loader(path) #transform Image into Tensor
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

"""
GET_XMP_COLOR_CLASS
    Input: image_path - root of image folder
    Output: N/A
    Comments:
        Open the files to write labels in labeled_images.txt and unlabeled_images.txt. 
        Read to XMP data and look for the photomechanic:ColorClass tag. Use this tag 
        as labels for the Missourian data. 
"""
def get_xmp_color_class(image_path):
    labels_file = open('labeled_images.txt', 'w')
    none_file = open('unlabeled_images.txt', 'w')

    for root, _, files in os.walk(image_path, topdown=True):
        for name in files:
            with open(os.path.join(root, name), 'rb') as f:
                img_str = str(f.read())
                xmp_start = img_str.find('photomechanic:ColorClass')
                xmp_end = img_str.find('photomechanic:Tagged')
                if xmp_start != xmp_end and xmp_start != -1:
                    xmp_str = img_str[xmp_start:xmp_end]
                    if xmp_str[26] != '0':
                        labels_file.write(xmp_str[26] + ', ' + str(os.path.join(root, name)))
                    else:
                        none_file.write(xmp_str[26] + ', ' + str(os.path.join(root, name)))
    
    labels_file.close()
    none_file.close()

# def get_xmp_color_class():
#     labels_file = open("labeled_images.txt", "w")
#     none_file = open("unlabeled_images.txt", "w")

#     data_loader = torch.utils.data.DataLoader(AdjustedDataset(data_dir, transform=transforms.Compose([transforms.ToTensor()])))

#     try:
#         for i, data in enumerate(data_loader):
#             if limit_num_pictures:
#                 if i > limit_num_pictures:
#                     break
#             _, _, path = data
#             path = path[0].rstrip()
#             try:
#                 with open(path, "rb") as f:
#                     img = f.read()
#                     img_string = str(img)
#                     xmp_start = img_string.find('photomechanic:ColorClass')
#                     xmp_end = img_string.find('photomechanic:Tagged')
#                     if xmp_start != xmp_end and xmp_start != -1:
#                         xmp_string = img_string[xmp_start:xmp_end]
#                         if xmp_string[26] != "0":
#                             print(xmp_string[26] + " " + str(path) + "\n\n")
#                             rated_indices.append(i)
#                             ratings.append(11 - int(xmp_string[26])) #have to invert and adjust to be on a growing scale of 1-10
#                             labels_file.write(xmp_string[26] + ", " + str(path) + ", " + str(i))
#                         else:
#                             ratings.append(0)
#                             bad_indices.append(i)
#                             none_file.write(xmp_string[26] + ", " + str(path) + ", " + str(i))
#             except OSError:
#                 logging.warning('Image #{} not found at specified path'.format(i))
#                 continue
#     except Exception as e:
#         logging.error('There was an error with the dataloader at image #{}: {}'.format(i, e))
#         sys.exit(1)
#     labels_file.close()
#     none_file.close()

'''
GET_MISSOURIAN_MAPPED_VAL
    Input: missourian raw val (int)
    Output: converted ranking
    Comments:
        Helper function to easily map raw 1-8 to 1-10 vals.
'''
def get_missourian_mapped_val(missourian_val):
    if(missourian_val == 1):
        return 10
    elif(missourian_val == 2):
        return 9
    elif(missourian_val == 3):
        return 7
    elif(missourian_val == 4):
        return 6
    elif(missourian_val == 5):
        return 5
    elif(missourian_val == 6):
        return 4
    elif(missourian_val == 7):
        return 2
    elif(missourian_val == 8):
        return 1
    else:
        return 0

"""
GET_FILE_COLOR_CLASS
    Input: N/A
    Output: pic_label_dict - (dict: path, label)
    Comments:
        Load Missourian labels from files, apply those to global arrays
"""
def get_file_color_class():
    pic_label_dict = {}
    try:
        labels_file = open("labeled_images.txt", "r")
        none_file = open("unlabeled_images.txt", "r")
    except OSError:
        logging.error('Could not open Missourian image mapping files')
        sys.exit(1)

    for line in labels_file:
        labels_string = line.split(',')
        # rated_indices.append(labels_string[0])
        file_name = (labels_string[1].split('/')[-1]).split('.')[0]
        pic_label_dict[file_name] = get_missourian_mapped_val(int(labels_string[0]))
        # ratings.append(get_missourian_mapped_val(int(labels_string[0])))

    for line in none_file:
        labels_string = line.split(',')
        # bad_indices.append(labels_string[0])
        file_name = (labels_string[1].split('/')[-1]).split('.')[0]
        pic_label_dict[file_name] = 0
        # ratings.append(0)

    logging.info('Successfully loaded info from Missourian Image Files')
    labels_file.close()
    none_file.close()
    return pic_label_dict

"""
GET_AVA_LABELS
    Input: N/A
    Output: dictionary mapping AVA file name to labels
    Comments:
        establish a blank dictionary and arbitarily high limit_lines count. Open the label file
        and read each of the results. Our resulting label is the largest number of votes recieved
        in the ranking. Return the dictionary containing the labels.
"""
def get_ava_labels():
    pic_label_dict = {}
    limit_lines = 1000000
    try:
        f = open(label_file, "r")
        for i, line in enumerate(f):
            if i >= limit_lines:
                logging.info('Reached the developer specified line limit')
                break
            line_array = line.split()
            picture_name = line_array[1]
            temp = line_array[2:]
            aesthetic_values = temp[:10]
            for i in range(0, len(aesthetic_values)): 
                aesthetic_values[i] = int(aesthetic_values[i])
            pic_label_dict[picture_name] = np.asarray(aesthetic_values).argmax()
        # logging.info('labels dictionary is {}'.format(pic_label_dict))
        logging.info('label dictionary completed')
        return pic_label_dict
    except OSError:
        logging.error('Cannot open AVA label file')
        sys.exit(1)

"""
BUILD_DATALOADERS
    Input: dataset
    Output:
        1. Training Dataloader
        2. Testing Dataloader
    Comments:
        Takes in the dataset identifier. Transforms the image to a 224 x 224 image to fit the vgg16 CNN.
        Completes a CenterCrop in attempt to maintain all features of images. Utilize the custom classes
        to supply paths and ratings with the dataloaders. Splits dataset according to the dataset.
        If it is the Missourian dataset split based on whether photo has existing rating. If it is AVA
        split randomly. Check the devices avaliable and choose GPU if possible, otherwise CPU. Finally
        return all dataloaders.

"""
def build_dataloaders(dataset, label_dict):
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
    train_data = AdjustedDataset(data_dir, label_dict, transform=_transform)
    test_data = AdjustedDataset(data_dir, label_dict, transform=_transform)
    logging.info('Training and Testing Dataset correctly transformed') 

    num_pictures = len(train_data)
    indices = list(range(num_pictures))
    split = int(np.floor(valid_size * num_pictures))
    if(dataset == 2):
        train_idx, test_idx = rated_indices, bad_indices
    else:
        train_idx, test_idx = indices[split:], indices[:split]

    # Define samplers that sample elements randomly without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Define data loaders, which allow batching and shuffling the data
    train_loader = torch.utils.data.DataLoader(train_data,
                sampler=train_sampler, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_data,
                sampler=test_sampler, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vgg16.to(device)
    logging.info('VGG16 is running on {}'.format(device))
    logging.info('train_loader is {}'.format(train_loader))
    return train_loader, test_loader

"""
CHANGE_FULLY_CONNECTED_LAYER
    Inputs: N/A
    Output: N/A
    Comments:
        1. Change the number of output classes to the model 
        2. Freeze all weights
        3. Remove the last layer
        4. Add on the new layer, 
            - switch 2nd number to the needed number of output classes
"""
def change_fully_connected_layer(): 
    vgg16.classifier[6].out_features = 10

    for param in vgg16.parameters():
        param.requires_grad = False
    logging.info('All VGG16 layers frozen')

    network = list(vgg16.classifier.children())[:-1]

    network.extend([nn.Linear(4096, 10)])
    vgg16.classifier = nn.Sequential(*network)
    logging.info('New Layer correctly added to VGG16')

"""
TRAIN_DATA_FUNCTION
    Inputs: 
        1. training dataloader
        2. Epoch count
        3. Previous Model File Name
        4. Dataset Type
        5. Label Dictionary (empty if AVA dataset)
        6. Final Model Name
    Outputs: N/A
    Comments:
        Will initially attempt to load in an existing model if supplied. 
        Then train depending on the given dataset. Optimizer and Loss Function is 
        set early with criterion and optimizer vars. Ends by saving the final 
        model out to the supplied file name.
"""
def train_data_function(train_loader, epochs, prev_model, dataset, label_dict, model_name):
    if(prev_model != 'N/A'):
        try:
            vgg16.load_state_dict(torch.load('../neural_net/models/' + prev_model))
        except Exception:
            logging.warning('Failed to find {}, model trained off base vgg16'.format(prev_model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=0.4, momentum=0.9)

    vgg16.train()
    training_loss = 0
    training_accuracy = 0
    num_epochs = epochs 

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        try:
            for i, data in enumerate(train_loader,0):
                if limit_num_pictures:
                    if i > limit_num_pictures:
                        break
                inputs, _, path, label = data
                if(dataset == 2):
                    try:
                        label = torch.LongTensor([int(label[0])])
                    except Exception:
                        logging.error('Invalid label for image, skipping Image #{} from label {}'.format(i, label))
                        continue
                else:
                    try:
                        path = path[0]
                        path_array = path.split('/')
                        pic_name = path_array[-1]
                        label = label_dict[pic_name.split('.')[0]]
                        label = torch.LongTensor([label])
                    except Exception:
                        logging.error('Invalid label found at path: {}, skipping Image #{}, label: {}'.format(path[0], i, label))
                        continue
                
                try:
                    optimizer.zero_grad()
                    output = vgg16(inputs)
                    loss = criterion(output, torch.LongTensor(label))
                    running_loss += loss.item()
                    _, preds = torch.max(output.data, 1)
                    num_correct += (preds == label).sum().item()
                    loss.backward()
                    optimizer.step()
                except Exception:
                    logging.warning('Issue calculating loss and optimizing with image #{}, dataloader is\n{}'.format(i, data))
                    continue
            
                if i % 2000 == 1999:
                    running_loss = 0
        except Exception:
            logging.error('Error reading epoch #{}, dumping data, saving backup model and exiting\n{}'.format(epoch, data))
            torch.save(vgg16.state_dict(), 'Backup_model.pt')
            sys.exit(1)

        training_loss = running_loss/len(train_loader.dataset)
        training_accuracy = 100 * num_correct/len(train_loader.dataset)
        print('training loss: {}\ntraining accuracy: {}'.format(training_loss, training_accuracy))
        #saving every epoch
        try:
            torch.save(vgg16.state_dict(), '../neural_net/models/' + model_name + '.pt')
        except Exception:
            logging.error('Unable to save model: {}, saving backup in root dir and exiting program'.format(model_name))
            torch.save(vgg16.state_dict(), 'Backup_model.pt')
            sys.exit(1)

"""
RUN
    Input: dataset identifier
    Output: N/A
    Comments:
        Gets AVA labels if AVA dataset or color classes if Missourian. After setting labels, 
        builds datasets and changes the final layer. Ends with training a new model.
        prompts user for previous model file name, epoch count, and model name file name.
"""
def run(dataset):
    logging.info('Begin running')
    label_dict = {}
    if(dataset == 'AVA' or dataset == '1'):
        dataset = 1
        label_dict = get_ava_labels()
        logging.info('Successfully loaded AVA labels')
    else:
        dataset = 2
        if(not path.exists('labeled_images.txt') or not path.exists('unlabeled_images.txt')):
            logging.info('labeled_images.txt and unlabeled_images.txt not found')
            get_xmp_color_class(data_dir)
        else:
            logging.info('labeled_images.txt and unlabeled_images.txt found')
        label_dict = get_file_color_class()
    train, test = build_dataloaders(dataset, label_dict)
    change_fully_connected_layer()
    train_data_function(train, epochs, prev_model, dataset, label_dict, model_name)
    # test_data_function(test)


"""
GLOBAL VARS
    Vars: 
        1. Dataset Path (based on chosen dataset)
        2. Labels Path
        3. Pictures Limit
        4. Rated Indicies Array
        5. Ratings Array
        6. Bad Indicies Array
        7. VGG16 Model
    Comments:
        Ends by calling the run function
"""
# root directory where the images are stored
dataset = 'AVA'
#prev_model = 'Jan31_All_2017_Fall_Dump_only_labels_10scale_and_AVA.pt'
prev_model = 'N/A'
logging.basicConfig(filename='logs/bkgd_quality_trainer.log', filemode='w', level=logging.DEBUG)
# model_name = 'Feb6_new_model_name.pt'
model_name = sys.argv[1]
if(model_name.split('.')[1] != 'pt'):
    logging.info('Invalid model name {} submitted, must end in .pt or .pth'.format(model_name))
    sys.exit('Invalid Model')
epochs = 25
if(dataset == 'AVA' or dataset == '1'):
    logging.info('Using AVA Dataset to train')
    data_dir = "/mnt/md0/reynolds/ava-dataset/images/"
else:
    logging.info('Using Missourian Dataset to train')
    data_dir = "/mnt/md0/mysql-dump-economists/Archives/2017/Fall/Dump"#/Fall"#/Dump"
    
label_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/AVA.txt"
tags_file = "/mnt/md0/reynolds/ava-dataset/AVA_dataset/tags.txt"
limit_num_pictures = False #limit_num_pictures = 2000
rated_indices = []
ratings = []
bad_indices = []
# we load the pretrained model, the argument pretrained=True implies to load the ImageNet weights for the pre-trained model
vgg16 = models.vgg16(pretrained=True)

#EXECUTE FILE
run(dataset)

