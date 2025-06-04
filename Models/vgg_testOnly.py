import cv2
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision import datasets, models
from torchvision.io import read_image, ImageReadMode
from torch.optim import lr_scheduler
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, precision_recall_curve
import seaborn as sb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg19_bn(pretrained = True)
for i, layer in enumerate(model.features):
    if i < 20:
        for param in layer.parameters():
            param.requires_grad = False

model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)

print(model)

torch.manual_seed(12)
torch.cuda.manual_seed(12)

class square_padding:
    def __call__(self, image):
        width, height = image.size
        max = np.max([width, height])
        height_pad = (max - height) // 2
        width_pad = (max - width) // 2
        padding = (width_pad, height_pad, width_pad, height_pad)
        return transforms.functional.pad(image, padding)

transform_format = transforms.Compose([
    square_padding(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# transform_aug = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(5),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transform_format
# ])

batch_size = 16

dataset = datasets.ImageFolder('Dataset_4_ImgOnly\\Dataset_Imgs', transform=transform_format)
train_dataset, val_test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
valid_dataset, test_dataset = train_test_split(val_test_dataset, test_size=0.5, random_state=42)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.to(device)

model.load_state_dict(torch.load("Models\\Saved Models\\VGG19\\best_vgg.pth")['state_dict'])
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

y_true, y_pred = [], []
test_loss, test_correct = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.float().unsqueeze(1).to(device)
        out = model(x)
        loss = criterion(out, y)
        test_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(out) > 0.5).float()
        test_correct += (preds == y).sum().item()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

test_acc = 100 * test_correct / len(test_loader.dataset)
test_loss /= len(test_loader.dataset)


precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

print(f"Test - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

cm = metrics.confusion_matrix(y_true, y_pred)
sb.heatmap(cm, annot=True, fmt='d', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
