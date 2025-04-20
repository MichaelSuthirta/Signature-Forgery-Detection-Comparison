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
transform_augment = transforms.RandomChoice([transforms.RandomRotation((5)), transforms.RandomHorizontalFlip()])
transform_multi_augment = transforms.RandomOrder([transforms.RandomRotation(5), transforms.RandomHorizontalFlip()])

transform_train_format = transforms.Compose([transforms.RandomChoice([transform_augment, transform_multi_augment], p=[0.25,0.15]), transform_format])

# Dataset preparation
# dataset = datasets.ImageFolder('Datasets', transform = transform_format)

batch = 32

train_dataset = datasets.ImageFolder('Datasets\Train', transform_train_format)
print("Train dataset processed.")

validate_dataset = datasets.ImageFolder('Datasets\Validate', transform_format)
print("Validation dataset processed.")

test_dataset = datasets.ImageFolder('Datasets\Test', transform_format)
print("Test dataset processed.")

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
        self.conv_layer_1 = nn.Conv2d(1,16,3)
        self.conv_layer_2 = nn.Conv2d(16,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc_1 = nn.Linear(64*34*34, 256)
        self.fc_2 = nn.Linear(256, 32)
        self.fc_3 = nn.Linear(32,2)
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.3)
    
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

# Early stopping
class EarlyStopping:
    def __init__(self, patience = 5, delta = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score == None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

epoch_amt = 100
learning_rate = 0.001

# Model
model = cnn().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training start. Device: {}".format(device))

early_stopping = EarlyStopping()

# Train and validation loss and accuracy arrays
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

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
    train_loss_avg = train_loss/len(train_data_load)

    print("Training - Epoch {}, Accuracy: {},  Loss: {}".format(epoch, train_accuracy, train_loss_avg))

    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss_avg)

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

    valid_loss_avg = valid_loss/len(validate_data_load)
    valid_accuracy = 100 * valid_correct / len(validate_dataset)
    print("Validation - Epoch {}, Accuracy: {},  Loss: {}".format(epoch, valid_accuracy, valid_loss_avg))

    valid_acc_list.append(valid_accuracy)
    valid_loss_list.append(valid_loss_avg)

    #Stop the training early
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Training stopped.")
        break

    valid_loss = 0.0

# Test loop
model.eval()

test_loss = 0.0
test_correct = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_data_load):
        images, labels = images.to(device), labels.to(device)
        prediction = model(images)
        test_loss += criterion(prediction, labels).item()

        test_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()

test_loss /= batch
test_accuracy = 100 * test_correct / len(test_dataset)

print("Test - Accuracy: {},  Loss: {}".format(test_accuracy, test_loss))

# Make train/validation graph
train_acc_list = torch.tensor(train_acc_list, device='cuda').cpu().numpy()
train_loss_list = torch.tensor(train_loss_list, device='cuda').cpu().numpy()
valid_acc_list = torch.tensor(valid_acc_list, device='cuda').cpu().numpy()
valid_loss_list = torch.tensor(valid_loss_list, device='cuda').cpu().numpy()

plt.plot(train_acc_list, label = "Train accuracy")
plt.plot(valid_acc_list, label = "Validation accuracy")
plt.legend()
plt.show()

plt.plot(train_loss_list, label = "Train loss")
plt.plot(valid_loss_list, label = "Validation loss")
plt.legend()
plt.show()