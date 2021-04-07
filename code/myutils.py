#%%
import os
from pathlib import Path

BASE_DIR = Path('.').resolve().parent
BASE_DIR

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
#%%

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def numbers_to_onehot(targets: list, n_classes: int):
    targets = np.array(targets).reshape(-1)
    one_hot_targets_2d = np.eye(n_classes)[targets]

    return one_hot_targets_2d

#%%
a = np.eye(6)
a


#%%