import torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import transform

import os
import time
import copy
import torchvision
from torchvision import transforms, utils, models

plt.ion()

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
            return None
        image = Image.open(img_name).convert('RGB')
        # classes = np.array([self.ava_frame.iloc[idx, 2:12]])
        classes_txt = self.ava_frame.iloc[idx, 12:14]
        classes = np.zeros((66))
        # print(classes_txt)
        for c in classes_txt:
            # print(c)
            if c != 0:
                classes[(int(c) - 1)] = 1
        classes = classes.astype('float').reshape(-1, 66)
        # print(classes)
        if self.transform:
            image = self.transform(image)
            classes = torch.from_numpy(classes)
            # print(classes)
        sample = {'image': image, 'classes': classes}

        return sample

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for batch in dataloader:
            inputs = batch['image'].to(device)
            labels = torch.flatten(batch['classes'].to(device), start_dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                _, ans  = torch.max(labels, 1)
                loss = criterion(outputs, ans)

                # backward + optimize only if in training phase
                if True:
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == ans)
        if True:
            scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "train", epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

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

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

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
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

ava_dataset = AVAImagesDataset(csv_file='/media/matt/New Volume/ava/cleanedlabels.txt', 
                               root_dir='/media/matt/New Volume/ava/ava-compressed/images/',
                               transform=transforms.Compose([
                                   transforms.Resize(256), 
                                   transforms.CenterCrop(224), 
                                   transforms.ToTensor(), 
                                   transforms.Normalize(
                                       mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                                       ]))

dataset_size = len(ava_dataset)

dataloader = DataLoader(ava_dataset, batch_size=32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Get a batch of training data
batch = next(iter(dataloader))
# print(batch['classes'])

# Make a grid from batch
out = torchvision.utils.make_grid(batch['image'])

model_ft = models.resnet50(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft.fc = nn.Linear(num_ftrs, 66)

model_ft.load_state_dict(torch.load("Dec1_r50_10ep_classifier.pt"))
for param in model_ft.parameters():
    param.requires_grad = True

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(model_ft.state_dict(), "Dec1_r50_25ep_full_classifier.pt")

visualize_model(model_ft)

imshow(out, title="class list")
