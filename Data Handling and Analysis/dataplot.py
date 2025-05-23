import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision import datasets
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

transform_train_format = transforms.Compose([transform_multi_augment, transform_format])

train_dataset = datasets.ImageFolder('Datasets\Train', transform_train_format)
print("Train dataset processed. Classes = {}".format(train_dataset.classes))

validate_dataset = datasets.ImageFolder('Datasets\Validate', transform_format)
print("Validation dataset processed. Classes = {}".format(validate_dataset.classes))

test_dataset = datasets.ImageFolder('Datasets\Test', transform_format)
print("Test dataset processed. Classes = {}".format(test_dataset.classes))


def check_dataset(dataset):
    # List image file names
    data_names = []
    for i in range(len(dataset)):
        data_names.append(os.path.basename(dataset.imgs[i][0])) # [0] means the file path, basename takes file name

    labels = []
    for i in range(len(dataset)):
        labels.append(dataset.imgs[i][1])

    correct_data = []
    wrong_data = []

    fake_counter = 0
    real_counter = 0

    for i in range(len(dataset)):
        if (data_names[i][0] == 'R' and labels[i] == 1) or (data_names[i][0] == 'F' and labels[i] == 0):
            correct_data.append(data_names[i])
            if labels[i] == 1:
                real_counter += 1
            else:
                fake_counter += 1
        else:
            wrong_data.append(data_names[i])

    print("Correct: {}, Wrong: {}".format(len(correct_data), len(wrong_data)))
    print("Fake: {}, Real: {}".format(fake_counter, real_counter))

    plt.bar(["Correct", "Wrong"], [len(correct_data), len(wrong_data)])
    plt.show()

    plt.bar(["Fake", "Real"], [fake_counter, real_counter])
    plt.show()

check_dataset(train_dataset)
check_dataset(validate_dataset)
check_dataset(test_dataset)

def check_data_amt(dataset):

    real = 0
    fake = 0

    print("Train data listing. Class index: {}".format(dataset.class_to_idx))
    for i in range(len(dataset)):
        if dataset.imgs[i][1] == 1:
            real += 1
        elif dataset.imgs[i][1] == 0:
            fake += 1
        # print(dataset.imgs[i])

    return real, fake

train_real, train_fake = check_data_amt(train_dataset)
valid_real, valid_fake = check_data_amt(validate_dataset)
test_real, test_fake = check_data_amt(test_dataset)

print(f"Train real: {train_real}, train fake: {train_fake}")
print(f"Valid real: {valid_real},  fake: {valid_fake}")
print(f"Test real: {test_real},  fake: {test_fake}")