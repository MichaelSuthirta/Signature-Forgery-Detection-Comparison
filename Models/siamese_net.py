import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision import datasets, models
from torchvision.io import decode_image
from torch.optim import lr_scheduler
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, precision_recall_curve
import seaborn as sb
import time

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

transform = transforms.Compose([square_padding(),
                               transforms.Resize((224,224)),
                               transforms.Grayscale(1),
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)]
                               )

dataset = pd.read_csv("Dataset_4_ori\\sign_data\\full_dataset.csv")
dataset.columns = ["Img 1", "Img 2", "Label"]

class ImgDataset():
    def __init__(self, set, dir, transform):
        self.dataset = set
        self.dir = dir
        self.transform = transform
    
    def __getitem__(self, index):
        img1_path = os.path.join(self.dir, self.dataset.iat[index,0])
        img2_path = os.path.join(self.dir, self.dataset.iat[index,1])
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        if self.transform != None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([int(self.dataset.iat[index,2])], dtype=np.float32))
    
    def __len__(self):
        return len(self.dataset)

# Split dataset into train, validation, and test with ratio of 8:1:1
train_dataset, val_test_set = train_test_split(dataset, test_size=0.2, random_state=42)
valid_dataset, test_dataset = train_test_split(val_test_set, test_size=0.5, random_state=42)

batch_size = 16

train_loader = DataLoader(dataset=ImgDataset(train_dataset, "Dataset_4_ori\\sign_data\\train", transform), batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=ImgDataset(valid_dataset, "Dataset_4_ori\\sign_data\\test", transform), batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=ImgDataset(test_dataset, "Dataset_4_ori\\sign_data\\test", transform), batch_size=batch_size, shuffle=False, num_workers=4)

# Define model, the highest performing one based on previous research is picked
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(86528, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    
    def forward_once(self, input):
        output = self.conv(input)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

save_model_path = os.path.join("Models\\Saved Models\\Siamese_Network", "best_siamese.pth")

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

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, y):
        diff = input1 - input2
        distance_square = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(distance_square)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * distance_square + (1-y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / input1.size()[0]
        return loss

def get_res(x1, x2, threshold=0.5):
    dist = torch.nn.functional.pairwise_distance(x1, x2)
    prediction = (dist<threshold).float()
    return prediction

if __name__ == "__main__":

    model = SiameseNet().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping()

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    epochs = 50

    print("Training starts.")

    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        train_total_data = 0
        for i, data in enumerate(train_loader, 0):
            img1, img2, label = data 
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label = label.squeeze()
            # print(f"Images in iteration {i} processed.")
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predict_result = get_res(out1, out2)
            train_correct += (predict_result == label).sum().item()
            train_total_data+=label.size(0)
        train_acc = 100 * train_correct / train_total_data
        train_loss /= len(train_loader.dataset)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        # train_loss = 0

        model.eval()
        val_loss, val_correct = 0, 0
        valid_total_data = 0
        with torch.no_grad():
            for i, data in enumerate(train_loader, 0):
                img1, img2, label = data 
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                label = label.squeeze()
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                val_loss += loss.item()
                predict_result = get_res(out1, out2)
                val_correct += (predict_result == label).sum().item()
                valid_total_data+=label.size(0)
        val_acc = 100 * val_correct / valid_total_data
        val_loss /= len(val_loader.dataset)

        val_accs.append(val_acc)
        val_losses.append(val_loss)

        # val_loss = 0

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% Train Loss={train_loss:.4f}, Validation Acc={val_acc: .2f}% Validation Loss={val_loss:.4f}")

        # save_model(model)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best(model)

    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    # model_path = torch.load("Models//Saved Models//Siamese_Network//best_siamese.pth")
    # model.load_state_dict(model_path['state_dict'])

    test_start_time = time.time()

    early_stopping.load_best(model)
    model.eval()

    y_true, y_pred = [], []
    test_loss, test_correct = 0, 0
    test_total_data = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            img1, img2, label = data 
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label = label.squeeze()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            test_loss += loss.item()
            # dist = torch.sqrt(torch.sum((img1-img2)**2, dim = 1))
            prediction = get_res(out1, out2)
            test_correct += (prediction==label).sum().item()
            test_total_data += label.size(0)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(prediction.cpu().numpy())

    test_end_time = time.time()

    test_time = test_end_time - test_start_time

    test_acc = 100 * test_correct / test_total_data
    test_loss /= len(test_loader.dataset)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    overall_end_time = time.time()

    overall_time = overall_end_time - overall_start_time

    print(f"Test - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Train accuracy: {train_time} s, test time: {test_time} s, overall time: {overall_time} s")

    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Siamese Network Accuracy")
    plt.show()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Siamese Network Loss")
    plt.show()

    cm = metrics.confusion_matrix(y_true, y_pred)
    sb.heatmap(cm, annot=True, fmt='d', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Siamese Network Confusion Matrix')
    plt.show()
