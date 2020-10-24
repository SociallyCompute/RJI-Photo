import logging, os.path, sys, torch, warnings

import sqlalchemy as sqla
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
sys.path.append(os.path.split(sys.path[0])[0])

from common import config
from common import connections

class DBNet(nn.Module):

    def __init__(self):
        super(DBNet, self).__init__()
        self.lin1 = nn.Linear(224, 224)

    def forward(self, x):
        return x