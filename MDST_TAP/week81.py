import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("Loading US Accidents dataset...")

file_path = 'data/us_accidents_data_week8.csv'
us_accidents = pd.read_csv(file_path).sample(frac=0.3, random_state=42)

def balance_data(df, target_column):
    target_counts = df[target_column].value_counts()
    majority_class = target_counts.idxmax()
    minority_classes = target_counts.index[target_counts.index != majority_class]

    df_majority = df[df[target_column] == majority_class]
    dfs_minority = [df[df[target_column] == minor] for minor in minority_classes]

    dfs_minority_oversampled = [resample(df_minor, replace=True, n_samples=target_counts.max(), random_state=42) for df_minor in dfs_minority]
    df_balanced = pd.concat([df_majority] + dfs_minority_oversampled)
    return df_balanced

us_accidents = balance_data(us_accidents, 'Severity')

print(f"Sample size after balancing: {len(us_accidents)}")

print("Preprocessing the dataset...")

us_accidents['Accident_Duration'] = (us_accidents['Accident_Duration'] - us_accidents['Accident_Duration'].mean()) / us_accidents['Accident_Duration'].std()

X = us_accidents[['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag', 'Distance(mi)', 
                  'Start_Hour_Sin', 'Start_Hour_Cos', 'Start_Month_Sin', 'Start_Month_Cos', 'Accident_Duration']].values
y = us_accidents['Severity'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y - 1, dtype=torch.long)

print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")

class_counts = np.bincount(y-1)
class_weights = torch.tensor(class_counts.max() / class_counts, dtype=torch.float32)

train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(list(zip(X_tensor, y_tensor)), [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("First few feature rows:\n", X_tensor[:5])
print("First few labels:\n", y_tensor[:5])

class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AccidentSeverityModel()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"Training samples: {train_size}, Test samples: {test_size}")

num_epochs = 40
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

min_test_loss = float('inf')
patience, trials = 3, 0

print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_targets = []

    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(targets.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    train_accuracy = accuracy_score(all_train_targets, all_train_preds)
    train_accuracies.append(train_accuracy)

    model.eval()
    total_test_loss = 0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    test_accuracy = accuracy_score(all_test_targets, all_test_preds)
    test_accuracies.append(test_accuracy)

    if avg_test_loss < min_test_loss:
        min_test_loss = avg_test_loss
        trials = 0
        torch.save(model.state_dict(), 'best_accident_severity_model.pth')
        print("Best model saved.")
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping!")
            break

    scheduler.step()

    report = classification_report(all_test_targets, all_test_preds, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'], output_dict=True)
    print(classification_report(all_test_targets, all_test_preds, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']))

    for severity, metrics in report.items():
        if isinstance(metrics, dict):
            precision = metrics["precision"]
            recall = metrics["recall"]
            if precision < 0.6 or recall < 0.6:
                print(f"Class {severity} failed to meet precision and recall thresholds.")
                # Consider adjusting model or data here

    print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(8,6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()