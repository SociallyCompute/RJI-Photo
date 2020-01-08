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
    # base_label = np.zeros(1000)
    counter = 0
    i = 0
    for _,label,paths in data_loader:
        i = i+1
        path=paths[0]
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

            '''
            these next if statements are flipping the values from a "color code" number to a ranking
            because we haven't flipped the final fully connected layer as of Dec 23, 2019 they are the range
            [000-999] where it is implied this is a decimal after the first number making the scale 0.00-9.99
            '''
            if(xmp_string[26] == '1'):
                label = torch.tensor([999])
            elif(xmp_string[26] == '2'):
                label = torch.tensor([800])
            elif(xmp_string[26] == '3'):
                label = torch.tensor([700])
            elif(xmp_string[26] == '4'):
                label = torch.tensor([650])
            elif(xmp_string[26] == '5'):
                label = torch.tensor([500])
            else:
                label = torch.tensor([250])
    print(counter)
    print("Total Images: " + str(i))