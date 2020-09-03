import logging, os.path, sys, torch, warnings

import matplotlib.pyplot as plt
import sqlalchemy as sqla
import torch.optim as optim
import numpy as np

from os import path
from PIL import ImageFile
from torch import nn
from torch import optim

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
sys.path.append(os.path.split(sys.path[0])[0])

from common import config
from common import connections

class ModelBuilder:

    """ ModelBuilder class makes a generalized model. It takes what it is given, 
        creates a dataset and provides testing/training functions for the model.

    :param model: (torchvision.model) model type (vgg, resnet, etc)
    :param custom_name: (string) name of the model when saving after 
        training or loading for testing
    :param batch: (int) batch size, 1 is online, otherwise its minibatch
    :param dataset: (string) value to identify dataset in usage (ava, missourian)
    :param subject: (string) content or quality flag
    :param device: (torch.cuda.device) string representing hardware being used
    :param outputs: (int) number of outputs expected from the custom model
    :param classification: (bool) flag denoting classification (True) or regression (False)
    """
    def __init__(self, model, custom_name, model_name, batch, dataset, subject, device, outputs, classification):
        
        self.model = model # base model
        self.custom_name = custom_name # custom name for our model
        self.model_name = model_name # string representation of base model name
        self.batch = batch #batch size
        self.dataset = dataset # dataset we are using
        self.pic_limit = None # flag for how many images to load
        self.rated_indices = [] # list to hold indices for images
        self.ratings = [] # list of image ratings
        self.bad_indices = [] # list to hold indices for images with zero labels
        self.train_data_samples = None
        self.test_data_samples = None
        self.subject = subject
        self.device = device
        self.outputs = outputs
        self.classification = classification
        
        self.db, self.photo_table = connections.make_db_connection('evaluation')

        if (dataset == 'ava'):
            logging.info('Using AVA Dataset')
            self.image_path = config.AVA_IMAGE_PATH
        else:
            logging.info('Using Missourian Dataset')
            self.image_path = config.MISSOURIAN_IMAGE_PATH

    def switch_fcc(self, freeze):
        """ Switch fully connected layer depending on the base model

        :param freeze: (String) describe whether or not to freeze all layers before the new layer
        """
        if freeze == 'freeze':
            for param in self.model.parameters():
                param.requires_grad = False
        if self.model_name == 'vgg':
            network = list(self.model.classifier.children())[:-1]
            network.extend([nn.Linear(4096, self.outputs)])
            network.extend([nn.Softmax()])
            self.model.classifier = nn.Sequential(*network)
        else: #resnet
            self.model.fc = nn.Sequential(
            nn.Linear(2048, self.outputs),
            #nn.Softmax()
            #nn.Sigmoid()
            )
        

    def to_device(self, data):
        """Switch data to model device

        :param data: ((list, tuple), Tensor, or nn.Layer) data to be converted to specified device
        """
        if isinstance(data, (list,tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)


    def evaluate(self, test_loader, outputs):
        """ Evaluate the testing set and save them to the database
        
        :param test_loader: (torch.utils.data.dataloader) dataloader containing the testing images
        :param outputs: (int) number of labels in the set (i.e. 10 for labels 1-10)
        """
        
        self.to_device(self.model)
        self.model.eval()
        ratings = []
        num_pictures = len(test_loader)
        index_progress = 0
        # logging.info('Running evaluation images in the test_loader of size: '
        #              '{}...'.format(len(test_loader)))

        while index_progress < num_pictures - 1:
            try:
                for i, (data, labels, index) in enumerate(test_loader, index_progress):

                    photo_path = self.test_data_samples[index_progress][0]

                    labels = torch.LongTensor(labels.to(self.device)).to(self.device)
                    data = data.to(self.device)

                    output = self.model(data)

                    # _, preds = torch.max(output.data, 1)
                    ratings = output[0].cpu().tolist()

                    # Prime tuples for database insertion
                    database_tuple = {}
                    for n in range(outputs):
                        database_tuple['model_score_{}'.format(n + 1)] = ratings[n]

                    # Include metadata for database tuple
                    database_tuple['photo_path'] = photo_path
                    database_tuple['photo_model'] = self.custom_name

                    # Insert tuple to database
                    result = self.db.execute(self.photo_table.insert().values(database_tuple))

            except Exception as e:
                logging.info('Ran into error for image #{}: {}\n... '
                             'Moving on.\n'.format(index_progress, e))
                
            index_progress += test_loader.batch
            
        # logging.info('Finished evaluation of images in the test_loader, the '
        #              'results are stored in the photo table in the database')


    def train(self, epochs, train_loader, prev_model, lr, mom, opt):
        """ Train model and save them as pytorch model. Save images 
            showing accuracy and loss functions over epochs.
            
        :param epochs:
        :param test_loader: (torch.utils.data.dataloader) dataloader 
            containing the testing images
        :param prev_model:    
        """
        if(prev_model != 'N/A'):
            try:
                self.model.load_state_dict(torch.load(config.MODEL_STORAGE_PATH + prev_model))
            except Exception:
                logging.warning(
                    'Failed to find {}, model trained off base resnet50'.format(prev_model))

        learning_rate = float(lr)
        mo = float(mom)

        self.model.train() #set model to training mode
        self.to_device(self.model) #put model on GPU

        if self.classification == 'True':
            criterion = nn.CrossEntropyLoss() #declare after all params are on device
        else: #regression
            criterion = nn.MSELoss() 

        if opt == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=mo) #declare after all params are on device
        elif opt == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        decay_rate = 0.95 #decay the lr each step to 95% of previous lr
        lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

        # self.model.train()
        training_loss = [0 for i in range(epochs)]
        training_accuracy = [0 for i in range(epochs)]
        num_pictures = len(train_loader) * self.batch
        logging.info('batches: {} num_pictures: {}'.format(len(train_loader), num_pictures))

        indices = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            num_correct = 0
            
            try:
                for i, (data, labels, index) in enumerate(train_loader,0):
                    if self.pic_limit:
                        if i > self.pic_limit:
                            break
                    try:
                        labels = labels.to(self.device)

                        if self.classification == 'True':
                            labels = torch.cuda.LongTensor(labels) if torch.cuda.is_available() else torch.LongTensor(labels)
                        else:
                            labels = torch.cuda.FloatTensor(labels.float()) if torch.cuda.is_available() else torch.FloatTensor(labels.float())

                        data = data.to(self.device)

                        output = self.model(data).to(self.device) #run model and get output
                        loss = criterion(output, labels) #calculate MSELoss given output and labels
                        optimizer.zero_grad() #zero all gradients in fully connected layer
                        loss.backward() #compute new gradients
                        optimizer.step() #update gradients
                        running_loss += loss.cpu().sum().item() #send loss tensor to cpu, then grab the value out of it
                        index_list = index.flatten().tolist()

                        #ensure each index_list is the batch size to correctly insert Postgres arrays
                        if len(index_list) != self.batch:
                            diff = self.batch - len(index_list)
                            for i in range(diff):
                                index_list.append(-1)
                        
                        indices.append(index_list)

                        if self.classification == 'True':
                            max_vals, prediction = torch.max(output.data, 1) 
                            corr = (prediction == labels).cpu().sum().item()
                            num_correct += corr #gets tensor from comparing predictions and labels, sends to cpu, sums tensor, grabs value out of it
                            logging.info('epoch:{} batch: {} accuracy: {} loss: {} total num_correct: {}'.format(epoch, 
                                i, (100 * (corr/self.batch)), loss.cpu().item(), num_correct))
                        else:
                            logging.info('epoch:{} batch: {} loss: {} total num_correct: {}'.format(epoch, 
                                i, loss.cpu().item(), num_correct))
                    except Exception as e:
                        logging.exception("""Issue calculating loss and optimizing with 
                                            image #{}, error is {}\ndata is\n{}""".format(i, e, data))
                        continue
                        
            except Exception:
                (data, label, index) = train_loader
                logging.error("""Error on epoch #{}, train_loader issue
                                with data: {}\nlabel: {}""".format(epoch, data, label))
                torch.save(self.model.state_dict(), self.custom_name)
                sys.exit(1)

            #values to insert into db
            db_tuple = {}
            db, train_table = connections.make_db_connection('training')
            db_tuple['te_dataset'] = self.dataset
            db_tuple['te_learning_rate'] = float(learning_rate[0]) if isinstance(learning_rate, (np.ndarray, list)) else float(learning_rate)
            db_tuple['te_momentum'] = mo
            db_tuple['te_model'] = self.model_name
            db_tuple['te_epoch'] = epoch
            db_tuple['te_batch_size'] = self.batch
            db_tuple['te_optimizer'] = 'adam'
            db_tuple['te_indices'] = indices
            indices = []
            training_loss[epoch] = running_loss/num_pictures
            db_tuple['te_loss'] = training_loss[epoch]

            if self.classification == 'True':
                training_accuracy[epoch] = 100 * (num_correct/num_pictures)
                db_tuple['te_accuracy'] = training_accuracy[epoch]
                logging.info('training loss: {}\ntraining accuracy: {}'.format(
                training_loss[epoch], training_accuracy[epoch]))
            else:
                logging.info('training loss: {}'.format(training_loss[epoch]))

            result = db.execute(train_table.insert().values(db_tuple))
            lr_sch.step() #decrease learning rate based on scheduler each epoch
            learning_rate = lr_sch.get_lr()

            # logging.info(torch.cuda.memory_summary())
            
            #final save of new model
            try:
                torch.save(self.model.state_dict(), config.MODEL_STORAGE_PATH + self.custom_name)
            except Exception:
                logging.error('Unable to save model: {}, '.format(self.custom_name),
                              'saving backup in root dir and exiting program')
                torch.save(self.model.state_dict(), self.custom_name)
                sys.exit(1)

        if self.classification == 'True':
            plt.figure(0)
            plt.plot([i for i in range(epochs)], training_accuracy)
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.title('Training Model Accuracy')
            plt.savefig('graphs/Train_Accuracy_' + self.custom_name[:-3] + '.png')

        plt.figure(1)
        plt.plot([i for i in range(epochs)], training_loss)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training Model Loss')
        plt.savefig('graphs/Train_Loss_' + self.custom_name[:-3] + '.png')
