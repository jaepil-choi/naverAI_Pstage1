#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
#%%

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

#%%
vgg = torchvision.models.vgg19_bn(pretrained=True)
vgg

#%%
# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes,):
        super().__init__()

        self.num_classes = num_classes
        self.net = torchvision.models.vgg19_bn(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        
        return x
#%%