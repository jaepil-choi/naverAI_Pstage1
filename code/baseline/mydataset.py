import os
from pathlib import Path

BASE_DIR = Path('.').resolve().parent
BASE_DIR

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

## custom libraries
import myutils

myutils.seed_everything(123)

n_classes_ALL = 18
n_classes_MASK = 3
n_classes_GENDER = 2

class MaskDataset(Dataset):
    def __init__(self, pkl_path, transform=None, target_transform=None):
        """Mask Dataset

        Args:
            pkl_path (str): Path of dataframe's pickle file
            transform (transforms, optional): Transform input data. Defaults to None.
            target_transform (transforms, optional): Transform output data. Defaults to None.
        """        
        self.df = pd.read_pickle(pkl_path)
        self.transform = transform
        self.target_transform = target_transform

        self.gender_encoded = np.eye(n_classes_GENDER)
        self.mask_encoded = np.eye(n_classes_MASK)
        self.cat_encoded = np.eye(n_classes_ALL)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx, :]
        img_name = row['path'] + row['filename']
        img_path = row['filepath']
        img_np = mpimg.imread(img_path) # H, W, C
        
        label_gender = row['gender_code']
        label_age = row['age']
        label_mask = row['mask_code']
        label_cat = row['cat_code']
        
        if self.transform:
            image = self.transform(image=img_np)
        if self.target_transform:
            pass
        
        sample = {
            'image': img_np, 
            'gender': label_gender, 
            'age': label_age, 
            'mask': label_mask, 
            'label': label_cat,
            }
        
        return sample

