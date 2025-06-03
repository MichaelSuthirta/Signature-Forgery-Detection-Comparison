import cv2
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

overall_start_time = time.time()

class square_padding:
    def __call__(self, image):
        width, height = image.size
        max = np.max([width, height])
        height_pad = (max - height) // 2
        width_pad = (max - width) // 2
        padding = (width_pad, height_pad, width_pad, height_pad)
        return transforms.functional.pad(image, padding)

if __name__ == "__main__":
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights
    # targets = [label for _, label in train_dataset.imgs]
    # counter = Counter(targets)
    # weights = [len(targets)/counter[i] for i in range(len(counter))]
    # class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    save_model_path = Path('Models/Saved Models/VGG19') / 'best_vgg.pth'

    class EarlyStopping:
        def __init__(self, patience=7, delta=0.001):
            self.patience = patience
            self.delta = delta
            self.best_score = None
            self.early_stop = False
            self.counter = 0
            self.best_model_state = None
            self.best_model_path = save_model_path

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None or score > self.best_score + self.delta:
                self.best_score = score
                self.best_model_state = model.state_dict()
                torch.save({'state_dict': model.state_dict(), 'loss': val_loss}, self.best_model_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        def load_best(self, model):
            best_record = torch.load(self.best_model_path)
            model.load_state_dict(best_record['state_dict'])
    
    # def save_model(model):
    #     torch.save({'state_dict': model.state_dict()}, save_model_path)

    model = models.vgg19_bn(pretrained=True)
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

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    early_stopping = EarlyStopping()
    epochs = 50

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            train_correct += (preds == y).sum().item()
        train_acc = 100 * train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.float().unsqueeze(1).to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(out) > 0.5).float()
                val_correct += (preds == y).sum().item()
        val_acc = 100 * val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% Train Loss={train_loss:.4f}, Validation Acc={val_acc: .2f}% Validation Loss={val_loss:.4f}")

        # save_model(model)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    early_stopping.load_best(model)

    test_start_time = time.time()
    model.eval()

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

    test_end_time = time.time()
    test_time = test_end_time - test_start_time

    test_acc = 100 * test_correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)


    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    overall_end_time = time.time()

    overall_time = overall_end_time-overall_start_time

    print(f"Test - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Train time: {train_time} s, test time: {test_time} s, overall time: {overall_time} s.")

    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

    cm = metrics.confusion_matrix(y_true, y_pred)
    sb.heatmap(cm, annot=True, fmt='d', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
