import cv2
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision import datasets, models
from torchvision.io import read_image
from torch.optim import lr_scheduler
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, precision_recall_curve
import seaborn as sb

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class square_padding:
    def __call__(self, image):
        width, height = image.size
        max = np.max([width, height])
        height_pad = (max - height) // 2

        width_pad = (max - width) // 2
        padding = (width_pad, height_pad, width_pad, height_pad)
        return pad(image, padding)        
    
if __name__ == "__main__":

    torch.manual_seed(12)
    torch.cuda.manual_seed(12)

    # Transform the images to be used
    transform_format = transforms.Compose([square_padding(), transforms.Resize((224,224)), transforms.Grayscale(3), 
                                           transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
    transform_multi_augment = transforms.RandomApply([transforms.RandomRotation(7), transforms.RandomHorizontalFlip(), 
                                                      transforms.RandomPerspective(distortion_scale=0.3), 
                                                      transforms.ColorJitter(brightness=0.3, contrast=0.3)])

    transform_train_format = transforms.Compose([transforms.RandomChoice([transforms.RandomOrder
                                                                          ([transform_multi_augment, transform_multi_augment]), 
                                                                          transform_multi_augment], p=[0.35, 0.35]), transform_format])

    # Dataset preparation

    batch = 32

    train_dataset = datasets.ImageFolder('New_Dataset\Train', transform_format)
    # y_train = train_dataset.classes
    print("Train dataset processed. Classes = {}".format(train_dataset.classes))

    # Imgs[0]: Image paths
    # Imgs[1]: Class index

    # image = plt.imread(train_dataset.imgs[0][0])
    # plot = plt.imshow(image)
    # plt.show()

    validate_dataset = datasets.ImageFolder('New_Dataset\Validate', transform_format)
    # y_valid = validate_dataset.classes
    print("Validation dataset processed. Classes = {}".format(validate_dataset.classes))

    test_dataset = datasets.ImageFolder('New_Dataset\Test', transform_format)
    # y_test = test_dataset.classes
    print("Test dataset processed. Classes = {}".format(test_dataset.classes))

    train_data_load = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    validate_data_load = DataLoader(validate_dataset, batch_size=batch, shuffle=False)
    test_data_load = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Early stopping
    class EarlyStopping:
        def __init__(self, patience = 5, delta = 0.001):
            self.patience = patience
            self.delta = delta
            self.best_score = None
            self.early_stop = False
            self.counter = 0
            self.best_model_state = None
            self.best_model_path = Path('Models\Saved Models\VGG19') / 'best_vgg.pth'
        
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

    model = models.vgg19_bn(pretrained = True)
    print(model.features)

    for i in range(6):
        for params in model.features[i].parameters():
            params.requires_grad = False

    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay= 1e-5)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.2)

    epoch_amt = 100

    model = model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("Training start. Device: {}".format(device))

    early_stopping = EarlyStopping()

    # Train and validation loss and accuracy arrays
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    # train_batch_acc = []
    # train_batch_loss = []
    # valid_batch_acc = []
    # valid_batch_loss = []

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
            result = output > 0.35
            # print(result)
            train_correct += (result == labels).float().sum()
            train_loss += loss.item() * images.size(0)
            # train_batch_loss.append(loss.item())
            # train_batch_acc.append((result == labels).float().sum())
            if(i == 1 or i % 50 == 0):
                print(f"Image {i} processed.")        

        train_accuracy = 100 * train_correct / len(train_dataset)
        # train_accuracy = train_correct / labels.size(0)
        train_loss_avg = train_loss/len(train_data_load.dataset)

        print(f"Training - Epoch {epoch}, Accuracy: {train_accuracy:.5f},  Loss: {train_loss_avg:.5f}")

        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss_avg)

        train_loss = 0.0

        # Validation
        valid_loss = 0.0
        valid_correct = 0

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(validate_data_load):
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                labels = labels.float()
                labels = labels.unsqueeze(1)
                loss = criterion(output, labels)

                # Calculate loss and accuracy
                # _, validate_result = torch.max(output, dim=1)
                validate_result = output > 0.35
                valid_correct += (validate_result == labels).float().sum()

                valid_loss += loss.item() * images.size(0)

                # valid_batch_loss.append(loss.item())
                # valid_batch_acc.append((validate_result == labels).float().sum())

        valid_loss_avg = valid_loss/len(validate_data_load.dataset)
        valid_accuracy = 100 * valid_correct / len(validate_dataset)
        # valid_accuracy = valid_correct / labels.size(0)

        scheduler.step()

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
        for i, (images, labels) in enumerate(test_data_load):
            images, labels = images.to(device), labels.to(device)
            true_labels.append(labels)
            prediction = model(images)
            labels = labels.float()
            labels = labels.unsqueeze(1)
            test_loss += criterion(prediction, labels).item()

            # test_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
            test_result = prediction > 0.35
            pred_labels.append(test_result)
            test_correct += ((test_result) == labels).type(torch.float).sum().item()

    true_labels = torch.cat(true_labels).cpu().numpy()
    pred_labels = torch.cat(pred_labels).cpu().numpy()
    test_loss /= batch
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