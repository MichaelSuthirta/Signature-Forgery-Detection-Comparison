import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision import datasets, models
from torchvision.io import read_image
from torch.optim import lr_scheduler
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, precision_recall_curve
import seaborn as sb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
        def __init__(self, patience = 6, delta = 0.0001):
            self.patience = patience
            self.delta = delta
            self.best_score = None
            self.early_stop = False
            self.counter = 0
            self.best_model_state = None
            self.best_model_path = Path('Models\Saved Models\CNN-AE') / 'best_cnn_ae.pth'
        
        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score == None:
                self.best_score = score
                self.best_model_state = model.state_dict()
                self.save_model(model, self.best_model_path, val_loss)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_model_state = model.state_dict()
                self.save_model(model, self.best_model_path, val_loss)
                self.counter = 0

        def save_model(self, model, modelPath, currentLoss):
            if(os.path.exists(modelPath) == False):
                torch.save({'state_dict': model.state_dict(), 'loss': currentLoss}, modelPath)
            else:
                saved_model = torch.load(modelPath, map_location=device, weights_only=True)
                if(currentLoss < saved_model['loss']):
                    torch.save({'state_dict': model.state_dict(), 'loss': currentLoss}, modelPath)
                else:
                    return

        def load_current_best_model(self, model):
            model.load_state_dict(self.best_model_state)
        
        def load_overall_best_model(self, model):
            best_record = torch.load(self.best_model_path)
            model.load_state_dict(best_record['state_dict'])

class cnn_ae(nn.Module):
    def __init__(self):
        super(cnn_ae, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8192, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = self.decoder(x)
        return x

batch_size = 32

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder("New_Datasets_Fixed\Datasets\Train", transform=transform)
valid_dataset = datasets.ImageFolder("New_Datasets_Fixed\Datasets\Validate", transform=transform)
test_dataset = datasets.ImageFolder("New_Datasets_Fixed\Datasets\Test", transform=transform)

train_load = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_load = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_load = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = cnn_ae().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
early_stopping = EarlyStopping()

threshold = 0.5

# Train and validation loss and accuracy arrays
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

for epoch in range(50):
        #Training
        train_loss = 0.0
        train_correct = 0

        model.train()
        for i, (images, labels) in enumerate(train_load):
            # Move image and label to device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(images)
            labels = labels.float()
            # print(labels)
            labels = labels.unsqueeze(1)
            loss = criterion(output, labels)

            # Normalization with L2 normalization
            # l2_normalize = sum(p.pow(2).sum() for p in model.parameters())
            # loss += 0.02 * l2_normalize

            # Backward propagation and optimization
            loss.backward()
            optimizer.step()

            # Loss and accuracy
            # _, result = torch.max(output, dim=1)
            # output = torch.sigmoid(output)
            result = output > threshold
            # print(result)
            train_correct += (result == labels).float().sum()
            train_loss += loss.item() * images.size(0)
            # train_batch_loss.append(loss.item())
            # train_batch_acc.append((result == labels).float().sum())
            # if(i == 1 or i % 50 == 0):
            #     print(f"Image {i} processed.")        

        train_accuracy = 100 * train_correct / len(train_dataset)
        # train_accuracy = train_correct / labels.size(0)
        train_loss_avg = train_loss/len(train_load.dataset)

        print(f"Training - Epoch {epoch}, Accuracy: {train_accuracy:.5f},  Loss: {train_loss_avg:.5f}")

        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss_avg)

        train_loss = 0.0

        # Validation
        valid_loss = 0.0
        valid_correct = 0

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_load):
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                labels = labels.float()
                labels = labels.unsqueeze(1)
                loss = criterion(output, labels)

                # Calculate loss and accuracy
                # _, validate_result = torch.max(output, dim=1)
                validate_result = output > threshold
                valid_correct += (validate_result == labels).float().sum()

                valid_loss += loss.item() * images.size(0)

                # valid_batch_loss.append(loss.item())
                # valid_batch_acc.append((validate_result == labels).float().sum())

        valid_loss_avg = valid_loss/len(valid_load.dataset)
        valid_accuracy = 100 * valid_correct / len(valid_dataset)
        # valid_accuracy = valid_correct / labels.size(0)


        print(f"Validation - Epoch {epoch}, Accuracy: {valid_accuracy:.5f},  Loss: {valid_loss_avg:.5f}")

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

true_labels = []
pred_labels = []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_load):
        images, labels = images.to(device), labels.to(device)
        true_labels.append(labels)
        prediction = model(images)
        labels = labels.float()
        labels = labels.unsqueeze(1)
        test_loss += criterion(prediction, labels).item()

        # test_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
        test_result = prediction > threshold
        pred_labels.append(test_result)
        test_correct += ((test_result) == labels).type(torch.float).sum().item()

true_labels = torch.cat(true_labels).cpu().numpy()
pred_labels = torch.cat(pred_labels).cpu().numpy()
test_loss /= batch_size
test_accuracy = 100 * test_correct / len(test_dataset)
test_precision = metrics.precision_score(true_labels, pred_labels, labels=[0,1])
test_recall = metrics.recall_score(true_labels, pred_labels, labels=[0,1])
test_f1 = metrics.f1_score(true_labels, pred_labels, labels=[0,1])
# test_accuracy = test_correct / labels.size(0)

print("Test - Accuracy: {},  Loss: {}, Precision: {}, Recall: {}, F1-Score: {}".format(test_accuracy, test_loss, test_precision, test_recall, test_f1))

# Make train/validation graph
train_acc_list = torch.tensor(train_acc_list, device='cuda').cpu().numpy()
train_loss_list = torch.tensor(train_loss_list, device='cuda').cpu().numpy()
valid_acc_list = torch.tensor(valid_acc_list, device='cuda').cpu().numpy()
valid_loss_list = torch.tensor(valid_loss_list, device='cuda').cpu().numpy()

# train_batch_loss= torch.tensor(train_batch_loss, device='cuda').cpu().numpy()
# train_batch_acc = torch.tensor(train_batch_acc, device='cuda').cpu().numpy()
# valid_batch_loss= torch.tensor(valid_batch_loss, device='cuda').cpu().numpy()
# valid_batch_acc = torch.tensor(valid_batch_acc, device='cuda').cpu().numpy()

plt.plot(train_acc_list, label = "Train accuracy")
plt.plot(valid_acc_list, label = "Validation accuracy")
plt.legend()
plt.show()

# plt.plot(train_batch_acc, label = "Train accuracy within all batches")
# plt.plot(valid_batch_acc, label = "Validation accuracy within all batches")
# plt.legend()
# plt.show()

plt.plot(train_loss_list, label = "Train loss")
plt.plot(valid_loss_list, label = "Validation loss")
plt.legend()
plt.show()

# plt.plot(train_batch_loss, label = "Train loss within all batches")
# plt.plot(valid_batch_loss, label = "Validation loss within all batches")
# plt.legend()
# plt.show()

# Accuracy metrics

matrix = confusion_matrix(true_labels, pred_labels, labels=[0,1])
sb.heatmap(matrix, annot=True, xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.title("Confusion Matrix", pad=15)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center')
plt.show()

# def train(model, train_loader, val_loader, epochs, device, filename="cnn_ae_best.pth"):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()

#     train_losses, val_losses = [], []
#     early_stopping = EarlyStopping(patience=5)

#     save_dir = "Models\Saved Models\CNN-AE"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, filename)

#     print(f"Model will be saved to: {save_path}")

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for inputs, _ in train_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, inputs)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         train_losses.append(avg_train_loss)

#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for val_inputs, _ in val_loader:
#                 val_inputs = val_inputs.to(device)
#                 val_outputs = model(val_inputs)
#                 val_loss += criterion(val_outputs, val_inputs).item()
#         avg_val_loss = val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)

#         print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
#         early_stopping(avg_val_loss, model)


#         if early_stopping.early_stop:
#             print("Training stopped")
#             break

#     print("Training complete.")
#     early_stopping.load_best_model(model)
#     torch.save(model.state_dict(), save_path)
#     print(f"Model saved to {save_path}")
#     return train_losses, val_losses



# def test_model(model, dataloader, threshold, device):
#     model.to(device)
#     model.eval()
#     criterion = nn.MSELoss()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for img, label in dataloader:
#             img = img.to(device)
#             output = model(img)
#             loss = criterion(output, img).item()
#             predicted = 1 if loss > threshold else 0
#             if predicted == label.item():
#                 correct += 1
#             print(f"Loss: {loss:.4f} | Predicted: {'Real' if predicted else 'Fake'}")

#     print(f"Accuracy: {correct}/{len(dataloader)}")

# # def validate_signatures(model, folder='Validate', threshold=0.01, device='cpu'):
# #     model.to(device)
# #     model.eval()
# #     transform = transforms.Compose([
# #         transforms.Grayscale(num_output_channels=1),
# #         transforms.Resize((128, 128)),
# #         transforms.ToTensor()
# #     ])

# #     print("\n--- Validating New Signatures ---")
# #     for root, _, files in os.walk(folder):
# #         for file in files:
# #             img_path = os.path.join(root, file)
# #             if not os.path.isfile(img_path) or not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
# #                 continue
# #             try:
# #                 img = Image.open(img_path).convert('L')
# #                 img = transform(img).unsqueeze(0).to(device)
# #                 with torch.no_grad():
# #                     output = model(img)
# #                     loss = F.mse_loss(output, img).item()
# #                     result = 'Fake' if loss > threshold else 'Real'
# #                     print(f"{img_path} - Reconstruction Error: {loss:.4f} -> {result}")
# #             except Exception as e:
# #                 print(f"Skipping {img_path} due to error: {e}")

# def plot_metrics(train_losses, val_losses=[], train_accuracies=[], val_accuracies=[]):
#     epochs = range(len(train_losses))

#     plt.figure(figsize=(8, 4))
#     plt.plot(epochs, train_losses, label='Train loss')
#     if len(val_losses) == len(train_losses):
#         plt.plot(epochs, val_losses, label='Validation loss')
#     else:
#         print(f"Skipping validation loss plot due to mismatched lengths: train={len(train_losses)}, val={len(val_losses)}")
#     plt.title("Loss over Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     if len(train_accuracies) == len(train_losses) and len(val_accuracies) == len(train_losses):
#         plt.figure(figsize=(8, 4))
#         plt.plot(epochs, train_accuracies, label='Train accuracy')
#         plt.plot(epochs, val_accuracies, label='Validation accuracy')
#         plt.title("Accuracy over Epochs")
#         plt.xlabel("Epoch")
#         plt.ylabel("Accuracy")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#     else:
#         print("Skipping accuracy plot due to mismatched lengths.")

# if __name__ == "__main__":
#     print("Current Working Directory:", os.getcwd())

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     batch_size = 32
#     epochs = 50
#     threshold = 0.01

#     train_loader, val_loader, test_loader = get_dataset(batch_size)
#     model = cnn_ae()

#     train_losses, val_losses = train(
#     model,
#     train_loader,
#     val_loader,
#     epochs=epochs,
#     device=device,
#     )
    
#     print("Testing the model.")
#     test_model(model, test_loader, 0.5, device)

#     # validate_signatures(model, folder='Validate', threshold=threshold, device=device)

#     plot_metrics(train_losses, val_losses)

