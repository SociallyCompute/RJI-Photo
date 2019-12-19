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
    counter = 0
    i = 0
    for _,label,paths in data_loader:
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
                label = torch.tensor(999)
            elif(xmp_string[26] == '2'):
                label = torch.tensor(800)
            elif(xmp_string[26] == '3'):
                label = torch.tensor(700)
            elif(xmp_string[26] == '4'):
                label = torch.tensor(650)
            elif(xmp_string[26] == '5'):
                label = torch.tensor(500)
            else:
                label = torch.tensor(250)
    print(counter)
    print("Total Images: " + str(i))