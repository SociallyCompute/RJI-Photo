from PIL import Image
import os, sys
import numpy as np
import pandas as pds

path_to_images = '/media/matt/New Volume/ava/ava-compressed/images'
path_to_labels = '/media/matt/New Volume/ava/AVA.txt'
# image_array = []
# label_array = []

def convert_dataset():
    labels = pds.read_csv(path_to_labels, delim_whitespace=True, header=None)
    labels = labels.set_index(1)
    print(labels)
    for _,f in enumerate(os.listdir(path_to_images)):
        idx = int(f.split('.')[0])
        img_array = np.array(Image.open(path_to_images + '/' + f))
        lab_array = np.array((labels.loc[idx])[1:11])
        np.savez('/media/matt/New Volume/ava/np_files/' + str(idx), x=img_array, y=lab_array)

convert_dataset()