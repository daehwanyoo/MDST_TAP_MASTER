import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report

# Load and filter data
print("Loading dataset...")
df = pd.read_csv("data/us_accidents_data_cleaned.csv")
df = df[df['Severity'].isin([1, 4])]

# Map Severity 1 -> 0, Severity 4 -> 1
df['Severity'] = df['Severity'].map({1: 0, 4: 1})
print(f"Filtered sample size: {len(df)}")

# Normalize numerical features
for col in ['Accident_Duration', 'Distance(mi)', 'Humidity(%)', 'Visibility(mi)']:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# One-hot encode categorical features
encoded = pd.get_dummies(df[['Weather_Condition', 'Sunrise_Sunset']], drop_first=False)
df = pd.concat([df, encoded], axis=1)

# Features and labels
feature_columns = ['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag', 
                   'Distance(mi)', 'Start_Hour_Sin', 'Start_Hour_Cos', 
                   'Start_Month_Sin', 'Start_Month_Cos', 'Accident_Duration', 
                   'Humidity(%)', 'Visibility(mi)'] + list(encoded.columns)

X = df[feature_columns].values.astype(np.float32)
y = df['Severity'].values.astype(np.int64)

# Convert to tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Train/test split
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

model = BinaryClassifier(X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("Starting training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    print(f"Epoch {epoch+1} | Train Accuracy: {acc:.4f}")

# Evaluation
print("Evaluating on test set...")
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

test_acc = accuracy_score(all_targets, all_preds)
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:")
print(classification_report(all_targets, all_preds, target_names=["Severity 1", "Severity 4"]))

torch.save(model.state_dict(), "severity_1_vs_4_model.pth")
