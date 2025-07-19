import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import numpy as np
from collections import Counter
import random

print("Loading US Accidents dataset...")

file_path = 'data/us_accidents_data_cleaned.csv'
us_accidents = pd.read_csv(file_path).sample(frac=0.3, random_state=42)
print(f"Sample size: {len(us_accidents)}")

print("Preprocessing the dataset...")

us_accidents['Accident_Duration'] = (us_accidents['Accident_Duration'] - us_accidents['Accident_Duration'].mean()) / us_accidents['Accident_Duration'].std()
us_accidents['Distance(mi)'] = (us_accidents['Distance(mi)'] - us_accidents['Distance(mi)'].mean()) / us_accidents['Distance(mi)'].std()

X = us_accidents[['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag', 'Distance(mi)', 
                  'Start_Hour_Sin', 'Start_Hour_Cos', 'Start_Month_Sin', 'Start_Month_Cos', 'Accident_Duration']].values
y = us_accidents['Severity'].values - 1

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")

class_counts = np.bincount(y)
class_weights = torch.tensor(class_counts.max() / class_counts, dtype=torch.float32)
print(f"Class Weights: {class_weights}")

dataset = list(zip(X_tensor, y_tensor))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

label_to_indices = {i: [] for i in range(4)}
for idx, (_, label) in enumerate(train_dataset):
    label_to_indices[label.item()].append(idx)

minority_targets = [0, 3]
max_len = max(len(label_to_indices[i]) for i in range(4))

augmented_indices = []
for label in range(4):
    indices = label_to_indices[label]
    if label in minority_targets:
        while len(indices) < max_len:
            indices += random.sample(label_to_indices[label], min(len(label_to_indices[label]), max_len - len(indices)))
    augmented_indices += indices

random.shuffle(augmented_indices)
train_dataset_upsampled = [train_dataset[i] for i in augmented_indices]

train_loader = DataLoader(train_dataset_upsampled, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("First few feature rows:\n", X_tensor[:5])
print("First few labels:\n", y_tensor[:5])

class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AccidentSeverityModel()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

print(f"Training samples: {len(train_dataset_upsampled)}, Test samples: {len(test_dataset)}")

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_preds, train_targets = [], []

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(targets.cpu().numpy())

    train_acc = accuracy_score(train_targets, train_preds)

    model.eval()
    total_test_loss = 0
    test_preds, test_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    pred_dist = pd.Series(test_preds).map({0: "Severity 1", 1: "Severity 2", 2: "Severity 3", 3: "Severity 4"}).value_counts(normalize=True)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print("Prediction Distribution:", pred_dist)
    print("Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=["Severity 1", "Severity 2", "Severity 3", "Severity 4"]))
