import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision import datasets
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create function to pad image (for image size)
class square_padding:
    def __call__(self, image):
        width, height = image.size
        max = np.max([width, height])
        height_pad = (max - height) / 2
        width_pad = (max - width) / 2
        padding = (width_pad, height_pad, width_pad, height_pad)
        return pad(image, padding)

# Transform the images to be used
transform_format = transforms.Compose([square_padding(), transforms.Resize(255), transforms.ToTensor(), transforms.Normalize(0.5,0.5)])

# Dataset preparation
dataset = datasets.ImageFolder('Datasets', transform = transform_format)

data_names = []

# List image file names
for i in range(len(dataset)):
    data_names.append(os.path.basename(dataset.imgs[i][0])) # [0] means the file path, basename takes file name

print(dataset.classes)

# Split dataset into train, validation and test with the ratio of 8:1:1
train_dataset, temp_dataset = train_test_split(dataset, train_size=0.8)
validate_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5)

train_data = DataLoader(train_dataset, batch_size=48, shuffle=True)
valid_data = DataLoader(validate_dataset, batch_size=48, shuffle=False)
test_data = DataLoader(test_dataset, batch_size=48, shuffle=False)

# Creating the CNN class
class cnn(nn.Module):
    def __init__(self):
        # super.__init__()
        # self.conv_layer_1 = nn.Conv2d(1,32,3,1)
        # self.conv_layer_2 = nn.Conv2d(32,64,3,1)
        # self.conv_layer_3 = nn.Conv2d(64,128,3,1)
        # self.pool = nn.MaxPool2d(2,2)
        # self.fc_1 = nn.Linear(, 128)
        # self.fc_2 = nn.Linear(128, 2)
        # self.dropout_1 = nn.Dropout2d(0.25)
        # self.dropout_2 = nn.Dropout2d(0.5)
        pass
    
    def forward(self, data):
        # data = self.pool(nn.functional.relu(self.conv_layer_1(data)))
        # data = self.pool(nn.functional.relu(self.conv_layer_2(data)))
        pass

epoch_amt = 100
learning_rate = 0.001

# Model
model = cnn().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_data)

for epoch in range(epoch_amt):
    for i, (images, labels) in enumerate(train_data):
        # Move image and label to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        loss = criterion(output, labels)

        # Backward propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()