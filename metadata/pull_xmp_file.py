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

def find_color_code(data_loader):
    # creating an array of 1000 inputs to map to the one hot encoding, 1 is high 0 is low
    base_label = np.zeros(1000)
    counter = 0
    i = 0
    for _,label,paths in data_loader:
        print(label)
        i = i+1
        path=paths[0]
        # print(path)
        with open(path, "rb") as f:
            img = f.read()
        img_string = str(img)
        xmp_start = img_string.find('photomechanic:ColorClass')
        xmp_end = img_string.find('photomechanic:Tagged')
        if xmp_start != xmp_end:
            xmp_string = img_string[xmp_start:xmp_end]
            if xmp_string[26] != "0":
                print(xmp_string[26] + " " + path + "\n\n")
            else:
                counter = counter + 1
            # labels[path.decode('ascii')] = xmp_string[26]
        # print(label)
            if(xmp_string[26] == '1'):
                base_label[999] = 1
                label = torch.tensor(base_label)
                base_label[999] = 0
            elif(xmp_string[26] == '2'):
                base_label[800] = 1
                label = torch.tensor(base_label)
                base_label[800] = 0
            elif(xmp_string[26] == '3'):
                base_label[700] = 1
                label = torch.tensor(base_label)
                base_label[700] = 0
            elif(xmp_string[26] == '4'):
                base_label[650] = 1
                label = torch.tensor(base_label)
                base_label[650] = 0
            elif(xmp_string[26] == '5'):
                base_label[500] = 1
                label = torch.tensor(base_label)
                base_label[500] = 0
            else:
                base_label[250] = 1
                label = torch.tensor(base_label)
                base_label[250] = 0
    print(counter)
    print("Total Images: " + str(i))