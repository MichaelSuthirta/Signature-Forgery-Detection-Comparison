import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transform the images to be used
transform_format = transforms.Compose([transforms.Resize(255), transforms.ToTensor()])

# Dataset preparation
dataset_x = datasets.ImageFolder('Datasets', transform = transform_format)
dataset_y = []

data_names = []

# List image file names
for i in range(len(dataset_x)):
    data_names.append(os.path.basename(dataset_x.imgs[i][0])) # [0] means the file path, basename takes file name

# Data class assigning
for i in range(len(dataset_x)):
    if data_names[i][0]=='R':
        dataset_y.append(1) # 1 for Real
    else:
        dataset_y.append(0) # 0 for Fake

for i in range(len(dataset_y)):
    print(dataset_y[i])

class SignatureDataset(Dataset):
    def __init__(self, directory, source, label):
        self.directory = directory
        self.source = source
        self.label = label
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        image = 

# dataset_x
# dataset_y = torch.tensor(dataset_y)

# print(type(dataset_x))
# print(type(dataset_y))

# dataset = TensorDataset(dataset_x, dataset_y)

# Split dataset into train, validation and testing dataset with ratio of 8:1:1
