import torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
from skimage import transform

import os
import time
import copy
import torchvision
from torchvision import transforms, utils, models

plt.ion()

"""
AVA classification
"""
# class AVAImagesDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.ava_frame = pd.read_csv(csv_file, sep=" ", header=None)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.ava_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # img_name = os.path.join(self.root_dir, str(self.ava_frame.iloc[idx, 0]) + '.jpg')
#         pre_string = '20170918_nurevabullyingprevention_nc_00' if (idx+1) < 10 else '20170918_nurevabullyingprevention_nc_0'
#         img_name = os.path.join(self.root_dir, pre_string + str(idx+1) + '.JPG')
        
#         if not os.path.isfile(img_name):
#             print(img_name + ' does not exist!')
#             return None
#         image = Image.open(img_name).convert('RGB')
#         classes = np.array([self.ava_frame.iloc[idx, 2:12]])
#         # classes_txt = self.ava_frame.iloc[idx, 12:14]
#         # classes = np.zeros((66))
#         # print(classes_txt)
#         # for c in classes_txt:
#             # print(c)
#             # if c != 0:
#                 # classes[(int(c) - 1)] = 1
#         classes = classes.astype('float').reshape(-1, 10)
#         # print(classes)
#         if self.transform:
#             image = self.transform(image)
#             classes = torch.from_numpy(classes)
#             # print(classes)
#         sample = {'image': image, 'classes': classes}

#         return sample

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
        self.num_outputs = num_outputs
        self.existing_model = existing_model
        self.resnet = models.resnet50(pretrained=True)
        if(frozen):
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_outputs)

    def forward(self, x):
        out = x
        return out

"""
OLD Train method
"""
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 1000.0

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         model.train()  # Set model to training mode

#         running_loss = 0.0
#         running_corrects = 0

#         # Iterate over data.
#         for batch in dataloader:
#             inputs = batch['image'].to(device)
#             labels = torch.flatten(batch['classes'].to(device), start_dim=1)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward
#             # track history if only in train
#             with torch.set_grad_enabled(True):
#                 outputs = model(inputs)
#                 output = torch.max(outputs)
#                 ans  = torch.max(labels).float()
#                 # print(output)
#                 # print(ans)
#                 loss = criterion(output, ans)

#                 # backward + optimize only if in training phase
#                 if True:
#                     loss.backward()
#                     optimizer.step()

#             # statistics
#             running_loss += loss.item() * inputs.size(0)
#             # running_corrects += torch.sum(preds == ans)
#         if True:
#             scheduler.step()

#         epoch_loss = running_loss / dataset_size
#         # epoch_acc = running_corrects.double() / dataset_size

#         print('{} Loss: {:.4f}'.format(
#             "train", epoch_loss))

#         # deep copy the model
#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             best_model_wts = copy.deepcopy(model.state_dict())

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best Loss: {:4f}'.format(best_loss))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

"""
NEW Train method
"""
def train_model(model,save_filepath,training_loader,validation_loader):

    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    data_loaders = {"train": training_loader, "val": validation_loader}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

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
    
                output = model(x)                              
                loss = loss_func(output, y)                  
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

# ava_dataset = AVAImagesDataset(csv_file='/media/matt/New Volume/ava/cleanedlabels.txt', 
#                                root_dir='/media/matt/New Volume/ava/ava-compressed/images',
#                                transform=transforms.Compose([
#                                    transforms.Resize(256), 
#                                    transforms.CenterCrop(224), 
#                                    transforms.ToTensor(), 
#                                    transforms.Normalize(
#                                        mean=[0.485, 0.456, 0.406], 
#                                        std=[0.229, 0.224, 0.225])
#                                        ]))

# dataset_size = len(ava_dataset)

# dataloader = DataLoader(ava_dataset, batch_size=1)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Get a batch of training data
# batch = next(iter(dataloader))
# # print(batch['classes'])

# # Make a grid from batch
# out = torchvision.utils.make_grid(batch['image'])

# model_ft = models.resnet50(pretrained=True)
# for param in model_ft.parameters():
#     param.requires_grad = False
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# # model_ft.fc = nn.Linear(num_ftrs, 10)
# model_ft.fc = nn.Linear(num_ftrs, 1)

# # model_ft.load_state_dict(torch.load("Nov17_r50_6ep_32batch.pt"))
# # for param in model_ft.parameters():
#     # param.requires_grad = True

# model_ft = model_ft.to(device)

# criterion = nn.MSELoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)
# torch.save(model_ft.state_dict(), "Dec10_r50_15ep_regression.pt")

# visualize_model(model_ft)

# # imshow(out, title="class list")

def get_dataset(data_filepath):
    data = np.load(data_filepath)

    tensor_x = torch.Tensor(data['x'])
    tensor_y = torch.Tensor(data['y'])

    dataset = TensorDataset(tensor_x,tensor_y)

    return dataset

if __name__ == "__main__":
    input_size = 8
    hidden_size = 64
    batch_first = True
    batch_size = 52
    # model = baselineLSTM(input_size,hidden_size,batch_size,batch_first)
    # model = baselineGRU(input_size,hidden_size,batch_size,batch_first)
    model = baselineResNet()
    # model = baselineFCNLSTM(input_size,hidden_size,batch_size,batch_first)

    training_dataset = get_dataset('eeg_dataset_training2.npz')
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)

    validation_dataset = get_dataset('eeg_dataset_testing2.npz')
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    # PATH = 'baselineLSTM.pth'
    # PATH = 'baselineGRU.pth'
    PATH = 'baselineRNN.pth'
    # PATH = 'baselineFCNLSTM.pth'
    train_model(model,PATH,training_loader,validation_loader)

