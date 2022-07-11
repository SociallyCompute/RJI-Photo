import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt

from tqdm import tqdm

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from old_codes.database import inserting
from models import architecture
from old_codes.config_files import paths

plt.ion()

"""
NEW Train method
"""
def train_model(model,save_filepath,training_loader,validation_loader):

    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    data_loaders = {"train": training_loader, "val": validation_loader}

    start_lr = 0.001
    m_type = "regression"
    num_outputs = 1
    epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    loss_func = nn.MSELoss() #nn.CrossEntropyLoss()

    model_id = inserting.insert_model(model, epochs, start_lr, m_type, num_outputs, loss_func, save_filepath)
    # training and testing
    for epoch in tqdm(range(100), position=0, leave=True):

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
                loss = loss_func(torch.squeeze(output).type(torch.FloatTensor), torch.squeeze(y).type(torch.FloatTensor))
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
        tqdm.write('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss, val_loss))
        loss_id = inserting.insert_loss(model_id, epoch, train_loss, val_loss)
    
    torch.save(model, save_filepath)

def get_np_dataset(data_filepath, idx):
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
    model = architecture.modifiedResNet(output_nodes)

    training_dataset = get_np_dataset('/media/matt/New Volume/ava/np_regress_files/',0)
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)

    validation_dataset = get_np_dataset('/media/matt/New Volume/ava/np_regress_files/',1024)
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    PATH = paths.MODEL_PATH
    train_model(model,PATH,training_loader,validation_loader)

