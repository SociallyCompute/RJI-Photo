import torch.nn as nn
from torchvision import models

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
