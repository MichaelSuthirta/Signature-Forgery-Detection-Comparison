import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import os
import random
import copy
from collections import Counter
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device in use: {device}')

data_path = '/content/drive/MyDrive/EGACNN/New_Dataset_TrainTest_2/Dataset_Imgs'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(data_path, transform=transform)
targets = [sample[1] for sample in dataset.samples]
train_idx, val_test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=targets, random_state=42)
val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, stratify=[targets[i] for i in val_test_idx], random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Train class distribution:", Counter([dataset[i][1] for i in train_idx]))
print("Validation class distribution:", Counter([dataset[i][1] for i in val_idx]))
print("Test class distribution:", Counter([dataset[i][1] for i in test_idx]))

def build_cnn(num_filters1=32, num_filters2=64, dropout_rate=0.5):
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(dropout_rate)
            self.features = nn.Sequential(
                nn.Conv2d(3, num_filters1, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters1, num_filters2, 3),
                nn.ReLU(),
                self.dropout,
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters2, 128, 3),
                nn.ReLU(),
                self.dropout,
                nn.MaxPool2d(2),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                dummy_out = self.features(dummy)
                self.flatten_dim = dummy_out.view(1, -1).shape[1]
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flatten_dim, 128),
                nn.ReLU(),
                self.dropout,
                nn.Linear(128, 2)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return CustomCNN()

def train_and_eval(model, return_history=False):
    model.to(device)
    all_labels = [dataset[i][1] for i in train_idx]
    class_counts = Counter(all_labels)
    weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(2)], dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    best_val_acc = 0
    patience, trigger_times = 7, 0
    best_model = None
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, train_correct = 0, 0
        val_loss_total, train_loss_total = 0, 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels)
                train_loss_total += loss.item() * inputs.size(0)

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)
                val_loss_total += loss.item() * inputs.size(0)

        history['train_acc'].append(train_correct.item() / len(train_loader.dataset))
        history['val_acc'].append(val_correct.item() / len(val_loader.dataset))
        history['train_loss'].append(train_loss_total / len(train_loader.dataset))
        history['val_loss'].append(val_loss_total / len(val_loader.dataset))

        if history['val_acc'][-1] > best_val_acc:
            best_val_acc = history['val_acc'][-1]
            best_model = copy.deepcopy(model)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    return (best_val_acc, best_model, history) if return_history else (best_val_acc, best_model)

random.seed(42)
POP_SIZE = 7
GENERATIONS = 4
TOP_K = 3

population = [
    (random.choice([16, 32, 64, 128]),
     random.choice([32, 64, 128, 256]),
     random.uniform(0.3, 0.6)) for _ in range(POP_SIZE)
]

top_models = []
gen_train_accuracies = []
gen_val_accuracies = []
gen_train_losses = []
gen_val_losses = []

for gen in range(GENERATIONS):
    print(f"\n=== Generation {gen+1} ===")
    results = []
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for config in population:
        model = build_cnn(*config)
        acc, trained_model, hist = train_and_eval(model, return_history=True)
        print(f"Config {config} => Val Acc: {acc:.4f}")
        results.append((acc, config, trained_model))

        train_accs.append(np.mean(hist['train_acc']))
        val_accs.append(np.mean(hist['val_acc']))
        train_losses.append(np.mean(hist['train_loss']))
        val_losses.append(np.mean(hist['val_loss']))

    gen_train_accuracies.append(np.mean(train_accs))
    gen_val_accuracies.append(np.mean(val_accs))
    gen_train_losses.append(np.mean(train_losses))
    gen_val_losses.append(np.mean(val_losses))

    results.sort(key=lambda x: x[0], reverse=True)
    top_models = results[:TOP_K]

    new_population = []
    while len(new_population) < POP_SIZE:
        p1, p2 = random.choice(top_models)[1], random.choice(top_models)[1]
        child = (
            random.choice([p1[0], p2[0]]),
            random.choice([p1[1], p2[1]]),
            min(0.7, max(0.3, (p1[2] + p2[2]) / 2 + random.uniform(-0.05, 0.05)))
        )
        new_population.append(child)
    population = new_population

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(gen_train_accuracies, label='Train Acc')
plt.plot(gen_val_accuracies, label='Val Acc')
plt.title('Accuracy per Generation')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(gen_train_losses, label='Train Loss')
plt.plot(gen_val_losses, label='Val Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\nEvaluating ensemble on test set...")
start_time = time.time()

y_true, y_pred_ensemble = [], []
all_preds = []

for _, _, model in top_models:
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            if len(y_true) < len(test_dataset):
                y_true.extend(labels.cpu().numpy())
    all_preds.append(preds)

for i in range(len(all_preds[0])):
    votes = [all_preds[m][i] for m in range(TOP_K)]
    vote_result = max(set(votes), key=votes.count)
    y_pred_ensemble.append(vote_result)

end_time = time.time()
test_duration = end_time - start_time

cm = confusion_matrix(y_true, y_pred_ensemble)
acc = 100. * sum([1 for i in range(len(y_true)) if y_true[i] == y_pred_ensemble[i]]) / len(y_true)
f1 = f1_score(y_true, y_pred_ensemble)
recall = recall_score(y_true, y_pred_ensemble)
prec = precision_score(y_true, y_pred_ensemble)

print(f"Ensemble Test Accuracy: {acc:.2f}%, Precision: {prec:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
print(f"Testing Duration: {test_duration:.2f} seconds")

sb.heatmap(cm, annot=True, fmt='d', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("EGACNN Confusion Matrix")
plt.show()