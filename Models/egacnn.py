import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.version
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

# print(device)
# print("torch.cuda.is_available():", torch.cuda.is_available())
# print("torch.version.cuda       :", torch.version.cuda)
# print("torch.cuda.device_count():", torch.cuda.device_count())
# print(torch.__version__)
# print(torch.version.cuda)

# Create function to pad image (for image size)
class square_padding:
    def __call__(self, image):
        width, height = image.size
        max = np.max([width, height])

        height_pad = (max - height) // 2

        width_pad = (max - width) // 2

        padding = (width_pad, height_pad, width_pad, height_pad)
        return pad(image, padding)

# Transform the images to be used
transform_format = transforms.Compose([square_padding(), transforms.Resize((144,144)), transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
# transform_augment = transforms.RandomApply([transforms.RandomRotation(15), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()])
# transform_augment_format = transforms.Compose([transform_augment, transform_format])

# Dataset preparation
# dataset = datasets.ImageFolder('Datasets', transform = transform_format)

batch = 64

train_dataset = datasets.ImageFolder('Datasets\Train', transform_format)
print(train_dataset.classes)
validate_dataset = datasets.ImageFolder('Datasets\Validate', transform_format)
print(validate_dataset.classes)
test_dataset = datasets.ImageFolder('Datasets\Test', transform_format)
print(test_dataset.classes)

train_data_load = DataLoader(train_dataset, batch_size=batch, shuffle=True)
validate_data_load = DataLoader(validate_dataset, batch_size=batch, shuffle=False)
test_data_load = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# data_names = []

# # List image file names
# for i in range(len(dataset)):
#     data_names.append(os.path.basename(dataset.imgs[i][0])) # [0] means the file path, basename takes file name

# print(dataset.classes)

# Creating the CNN class
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv_layer_1 = nn.Conv2d(1,8,3)
        self.conv_layer_2 = nn.Conv2d(8,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc_1 = nn.Linear(16*34*34, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64,2)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.5)
    
    def forward(self, data):
        data = self.pool(nn.functional.relu(self.conv_layer_1(data)))
        data = self.pool(nn.functional.relu(self.conv_layer_2(data)))
        data = torch.flatten(data, 1)
        data = nn.functional.relu(self.fc_1(data))
        data = self.dropout_1(data)
        data = nn.functional.relu(self.fc_2(data))
        data = self.dropout_2(data)
        data = self.fc_3(data)
        return data

epoch_amt = 100
learning_rate = 0.001

# Model
model = cnn().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training start. Device: {}".format(device))

# Training and validation loop
for epoch in range(epoch_amt):
    #Training
    train_loss = 0.0
    train_correct = 0

    model.train()
    for i, (images, labels) in enumerate(train_data_load):
        # Move image and label to device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(images)
        loss = criterion(output, labels)

        # Backward propagation and optimization
        loss.backward()
        optimizer.step()

        # Loss and accuracy
        _, result = torch.max(output, dim=1)
        train_correct += (result == labels).float().sum()
        train_loss += loss.item()

    train_accuracy = 100 * train_correct / len(train_dataset)

    print("Training - Epoch {}, Accuracy: {},  Loss: {}".format(epoch, train_accuracy, train_loss))
    train_loss = 0.0

    # Validation
    valid_loss = 0.0
    valid_correct = 0

    model.eval()
    for i, (images, labels) in enumerate(validate_data_load):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        # Calculate loss and accuracy
        _, validate_result = torch.max(output, dim=1)
        valid_correct += (validate_result == labels).float().sum()

        valid_loss += loss.item()

    valid_accuracy = 100 * valid_correct / len(validate_dataset)
    print("Validation - Epoch {}, Accuracy: {},  Loss: {}".format(epoch, valid_accuracy, valid_loss))
    valid_loss = 0.0

# # Test loop
# model.eval()

# test_loss = 0.0
# test_correct = 0

# with torch.no_grad():
#     for i, (images, labels) in enumerate(test_data_load):
#         prediction = model(images)
#         test_loss += criterion(prediction, labels).item()

#         test_correct += (prediction.argmax(1) == labels)