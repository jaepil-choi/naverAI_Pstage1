#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
#%%

#%%
class MyModel(nn.Module):
    def __init__(self, num_classes,):
        super().__init__()

        self.model = torchvision.models.vgg19_bn(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        self.model(x)
        
        return x