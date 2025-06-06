import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import random
import copy
from collections import Counter

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training start. Device: {device}')

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset loading
train_dataset = datasets.ImageFolder('New_Dataset/Train', transform=transform_train)
val_dataset = datasets.ImageFolder('New_Dataset/Validate', transform=transform_val_test)
test_dataset = datasets.ImageFolder('New_Dataset/Test', transform=transform_val_test)

print(f'Train dataset processed. Classes = {train_dataset.classes}')
print(f'Validation dataset processed. Classes = {val_dataset.classes}')
print(f'Test dataset processed. Classes = {test_dataset.classes}')

print("Train class distribution:", Counter([label for _, label in train_dataset]))
print("Validation class distribution:", Counter([label for _, label in val_dataset]))
print("Test class distribution:", Counter([label for _, label in test_dataset]))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN class
def build_cnn(num_filters1=32, num_filters2=64, dropout_rate=0.5):
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, num_filters1, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(num_filters2, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.dropout = nn.Dropout(dropout_rate)

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

# Train and evaluate function
def train_and_eval(model):
    model.to(device)
    class_counts = Counter(train_dataset.targets)
    weights = torch.tensor([1.0 / class_counts[i] for i in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)  # Increased learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    best_val_acc = 0
    patience, trigger_times = 7, 0
    best_model = None

    for epoch in range(60):  # Increased epochs
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels)
        acc = correct.item() / len(val_loader.dataset)
        scheduler.step()

        if acc > best_val_acc:
            best_val_acc = acc
            best_model = copy.deepcopy(model)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    return best_val_acc, best_model

# Genetic algorithm setup
POP_SIZE = 10
GENERATIONS = 8
TOP_K = 3

random.seed(22)

# Generate initial population
POP_SIZE = 7
GENERATIONS = 4
TOP_K = 4

random.seed(42)
population = [
    (random.choice([16, 32, 64, 128]),
     random.choice([32, 64, 128, 256]),
     random.uniform(0.3, 0.6)) for _ in range(POP_SIZE)
]

top_models = []
for gen in range(GENERATIONS):
    print(f"\n=== Generation {gen+1} ===")
    results = []
    for config in population:
        model = build_cnn(*config)
        acc, trained_model = train_and_eval(model)
        print(f"Config {config} => Val Acc: {acc:.4f}")
        results.append((acc, config, trained_model))

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

# Evaluation
print("\nEvaluating ensemble on test set...")
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
    all_preds.append(preds)

for i in range(len(all_preds[0])):
    votes = [all_preds[m][i] for m in range(TOP_K)]
    vote_result = max(set(votes), key=votes.count)
    y_pred_ensemble.append(vote_result)
    y_true.append(test_dataset.targets[i])

cm = confusion_matrix(y_true, y_pred_ensemble)
acc = 100. * sum([1 for i in range(len(y_true)) if y_true[i] == y_pred_ensemble[i]]) / len(y_true)
f1 = f1_score(y_true, y_pred_ensemble, labels=[0,1])
recall = recall_score(y_true, y_pred_ensemble, labels=[0,1])
prec = precision_score(y_true, y_pred_ensemble, labels=[0,1])

print(f"Ensemble Test Accuracy: {acc:.2f}%, Precision: {prec:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes).plot()
plt.title("EGACNN Confusion Matrix on Test Data")
plt.show()