import logging, os, os.path, sys, torch, warnings

import torchvision.models as models

from os import path
from PIL import ImageFile
from torch import nn

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
sys.path.append(os.path.split(sys.path[0])[0])

from common import model as mo
from common import config
from common import image_processing as im

def run_train_model(model_name, custom_model, epochs, outputs, device):
    """ Prepare the data and run the model in training mode

    :param model_name: (String) name describing base model
    :param custom_model: (mo.ModelBuilder) custom ModelBuilder class
    :param epochs: (int) number of times to train the model on the dataset
    :param outputs: (int) number of expected outputs
    :param device: (String ("cpu" or "cuda:0")) describe what hardware is running
    """
    # AVA
    if(custom_model.dataset == 'ava'): 
        label_dict = im.get_ava_quality_labels(custom_model.pic_limit)
        # logging.info(label_dict)

    # Missourian
    elif(custom_model.dataset == 'missourian'): 
        if(not path.exists('Mar13_labeled_images.txt') or \
           not path.exists('Mar13_unlabeled_images.txt')):
            logging.info('labeled_images.txt or unlabeled_images.txt not found')
            im.get_xmp_color_class(custom_model.rated_indices, custom_model.bad_indices)
        else:
            logging.info('labeled_images.txt and unlabeled_images.txt found')
        label_dict = custom_model.get_file_color_class()
    else: #classifier
        label_dict = custom_model.get_classifier_labels()

    train, _ = im.build_dataloaders(custom_model, label_dict)
    custom_model.switch_fcc(freeze)
    custom_model.train(epochs, train, custom_name, lr, momentum, optimizer)


def run_test_model(model_name, custom_model, outputs, device):
    """ Test models
    
    :param model_name: (string) identify which type of model is being run
    :param custom_model: (ModelBuilder) custom ModelBuilder base class
    :param outputs: (int) number of output layers on final fully connected layer
    :param device: (String ("cpu" or "cuda:0")) describe what hardware is running
    """
    logging.info('Begin running')
    label_dict = {}
    label_dict = custom_model.get_file_color_class()
    _, test = im.build_dataloaders(custom_model, label_dict)
    custom_model.switch_fcc(freeze)

    custom_model.evaluate(test, outputs)

"""
Basic Script Functions, defines global vars and starts functions
"""

custom_name = sys.argv[1]
dataset = sys.argv[2]
epochs = int(sys.argv[3])
batch = int(sys.argv[4])
model_name = sys.argv[5] # 'vgg16' or 'resnet'
subject = sys.argv[6] # 'content' or 'quality'
freeze = sys.argv[7]
lr = sys.argv[8]
momentum = sys.argv[9]
optimizer = sys.argv[10]
classification = sys.argv[11]
test = sys.argv[12]

logging.basicConfig(filename='logs/' + custom_name + '.log', 
                    filemode='w', level=logging.DEBUG)
custom_name = custom_name + '.pt'

if subject == 'quality':
    outputs = 1
elif subject == 'content':
    outputs = 67
else:
    logging.info('The classification subject you specified ({}) '.format(subject),
                'does not exist, please choose from \'quality\' or \'content\'\n')
    sys.exit(1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if model_name == 'vgg16':
    model = models.vgg16(pretrained=True).to(device)
elif model_name == 'resnet':
    model = models.resnet50(pretrained=True).to(device)
else:
    logging.info('Invalid model requested: {}. '.format(model),
                'Please choose from \'vgg16\' or \'resnet\'\n')
    sys.exit('Invalid Model')

custom_model = mo.ModelBuilder(model, custom_name, model_name, batch, dataset, subject, device, outputs, classification)

if os.path.isfile(config.MODEL_STORAGE_PATH + custom_name) and test == '1':
    logging.info('Running Model in Testing Mode')
    run_test_model(model_name, custom_model, outputs, device)
else:
    logging.info('Running Model in Training Mode')
    run_train_model(model_name, custom_model, epochs, outputs, device)
    





