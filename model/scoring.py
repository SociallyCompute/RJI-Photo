import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from database import inserting
from model import architecture

def score_individual_img(img_path):
    with Image.open(str(img_path)) as img:
        img = img.convert('RGB')
        transform = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                        ])
        img_array = np.array(transform(img))
    return(score_img(img_array, img_path))

def score_img(np_img,img_path):
    PATH = 'model_files/modifiedResNet.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_nodes = 1
    model = architecture.modifiedResNet(output_nodes)
    model.load(PATH)
    model.training(False)
    model.to(device)
    x = torch.Tensor(np_img).to(device)
    score = int(model(x).cpu().numpy()[0])
    inserting.insert_photos(img_path, score)
    return score

def score_folder_imgs(folder_path):
    score_dict = {}
    for img_path in os.listdir(str(folder_path)):
        if (img_path.endswith(".png") or 
            img_path.endswith(".PNG") or 
            img_path.endswith(".jpg") or 
            img_path.endswith(".JPG")
        ):
            img = os.path.join(folder_path, img_path)
            score_dict[img_path] = score_individual_img(img)
    return score_dict