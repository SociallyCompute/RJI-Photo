import torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
from skimage import transform
from sklearn.metrics import accuracy_score

import os
import time
import copy
import torchvision
from torchvision import transforms, utils, models

plt.ion()

"""
AVA Regression
"""
class AVAImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.ava_frame = pd.read_csv(csv_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ava_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.ava_frame.iloc[idx, 0]) + '.jpg')
        
        if not os.path.isfile(img_name):
            print(img_name + ' does not exist!')
            return None
        image = Image.open(img_name).convert('RGB')
        regress = self.ava_frame.iloc[idx, 2:12].values.tolist()
        result = 0
        for i in range(len(regress)):
            result += i*regress[i]
        result = result/sum(regress)
        regress = np.array([result])
        regress = regress.astype('float').reshape(-1, 1)
        if self.transform:
            image = self.transform(image)
            regress = torch.from_numpy(regress)
        sample = {'image': image, 'classes': regress}

        return sample

"""
This is done exclusively to have a baseline model for ResNet for verification purposes
"""
class baselineResNet(nn.Module):
    def __init__(self,pretrain=True,frozen=True):
        super(baselineResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrain)
        if(frozen):
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.resnet(x)
        return out

"""
Our modified version with changing the final layer (using CNN as classifier)
"""
class modifiedResNet(nn.Module):
    def __init__(self,num_outputs,existing_model=None,frozen=False):
        super(modifiedResNet, self).__init__()
        self.num_outputs = num_outputs
        self.existing_model = existing_model
        self.resnet = models.resnet50(pretrained=True)
        if(frozen):
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_outputs)

    def forward(self, x):
        out = self.resnet(x)
        return out

"""
NEW Train method
"""
def train_model(model,save_filepath,training_loader,validation_loader):

    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    data_loaders = {"train": training_loader, "val": validation_loader}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss() #nn.CrossEntropyLoss()

    # training and testing
    for epoch in range(100):

        train_loss = 0.0
        val_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):       
                # print(x.dtype)
                output = model(x)   
                # print(torch.squeeze(output))
                # print(y)  
                # print("output: " + str(torch.argmax(output, dim=1)))
                # print(accuracy_score(output.detach().cpu().numpy(), y.detach().cpu().numpy()))
                # print("labels: " + str(torch.argmax(y, dim=1)))
                # loss = loss_func(output, torch.argmax(y, dim=1))
                loss = loss_func(torch.squeeze(output).type(torch.FloatTensor), torch.argmax(y).type(torch.FloatTensor))
                # print(loss.dtype)
                optimizer.zero_grad()           

                if phase == 'train':
                    loss.backward()
                    optimizer.step()                                      

                running_loss += loss.item()
            
            if phase == 'train':
                train_loss = running_loss
            else:
                val_loss = running_loss

        # shows average loss
        # print('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss/training_len, val_loss/validation_len))
        # shows total loss
        print('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss, val_loss))
    
    torch.save(model, save_filepath)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show(block=True)

def visualize_model(model, num_images=64):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    # count = {i: 0 for i in range(1, 11)}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['image'].to(device)
            labels = batch['classes'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j] + 1))
                # print((preds[j]+1).cpu().item())
                # count[(preds[j]+1).cpu().item()] += 1
                # imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # print(count)
                    return
        model.train(mode=was_training)

def get_dataset(data_filepath, idx):
    dataset_list = []
    for i,f in enumerate(os.listdir(data_filepath), idx):
        data = np.load(data_filepath + f)
        if (i - idx) > 1023:
            break
        tensor_x = torch.Tensor(data['x'])
        tensor_x = torch.unsqueeze(tensor_x, 0)
        # print(tensor_x.size())
        tensor_y = torch.Tensor(data['y'])
        tensor_y = torch.unsqueeze(tensor_y, 0)
        # print(tensor_y.size())

        dataset_list.append(TensorDataset(tensor_x,tensor_y))

    dataset = ConcatDataset(dataset_list)
    return dataset

if __name__ == "__main__":
    batch_size = 32
    output_nodes = 1
    model = modifiedResNet(output_nodes)

    training_dataset = get_dataset('/media/matt/New Volume/ava/np_regress_files/',0)
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)

    validation_dataset = get_dataset('/media/matt/New Volume/ava/np_regress_files/',1024)
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    PATH = 'modifiedResNet.pt'
    train_model(model,PATH,training_loader,validation_loader)

