from PIL import Image
import os, sys
import numpy as np
import pandas as pds
from torchvision import transforms

path_to_images = '/media/matt/New Volume/ava/ava-compressed/images'
path_to_labels = '/media/matt/New Volume/ava/AVA.txt'
# image_array = []
# label_array = []

def convert_dataset():
    labels = pds.read_csv(path_to_labels, delim_whitespace=True, header=None)
    labels = labels.set_index(1)
    print(labels)
    for _,f in enumerate(os.listdir(path_to_images)):
        try:
            idx = int(f.split('.')[0])
            img = Image.open(path_to_images + '/' + f)
            img = img.convert('RGB')
            transform = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Normalize(
                            #     mean=[0.5],
                            #     std=[0.5])
                            # ])
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                            ])
            img_array = np.array(transform(img))
            lab_array = np.array((labels.loc[idx])[1:11])
            result = 0
            for i in range(len(lab_array)):
                result += i*lab_array[i]
            result = result/np.sum(lab_array)
            lab_array = np.array(result)
            np.savez('/media/matt/New Volume/ava/np_regress_files/' + str(idx), x=img_array, y=lab_array)
        except Exception as ex:
            print(f)
            print(ex)

convert_dataset()