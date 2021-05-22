from logging import root
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from database import inserting
from model import architecture
from config_files import paths

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
    PATH = paths.MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_nodes = 1
    model = architecture.modifiedResNet(output_nodes)
    model = torch.load(PATH)
    model.eval()
    model.to(device)
    x = torch.unsqueeze(torch.Tensor(np_img).to(device),0)
    score = int(model(x).cpu().detach().numpy()[0])
    inserting.insert_photos(img_path, score)
    return score

def score_folder_imgs(folder_path):
    score_dict = {}
    for root,dirs,f in os.walk(folder_path):
        for file in f:
            if (file.endswith(".png") or 
                file.endswith(".PNG") or 
                file.endswith(".jpg") or 
                file.endswith(".JPG")
            ):
                img = os.path.join(root, file)
                score_dict[img] = score_individual_img(img)
    return score_dict

def score_all():
    return score_folder_imgs('/media/matt/foo-fighter/reyndolds/')

if __name__ == "__main__":
    # print(score_folder_imgs(paths.MISSOURIAN_IMAGE_PATH))
    score_all()