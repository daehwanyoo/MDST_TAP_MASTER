import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

print("Loading US Accidents dataset...")
us_accidents = pd.read_csv('US_Accidents_March23.csv')
print("US Accidents dataset loaded.")
print(us_accidents.head())

print("Dataset Info:")
us_accidents.info()

print("Performing Exploratory Data Analysis...")
plt.figure(figsize=(10, 6))
sns.countplot(data=us_accidents, x='Severity', order=us_accidents['Severity'].value_counts().index)
plt.title('Accident Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Count')
plt.show()

us_accidents['Start_Time'] = us_accidents['Start_Time'].str.split('.').str[0]
us_accidents['Start_Time'] = pd.to_datetime(us_accidents['Start_Time'].str.split('.').str[0], errors='coerce')
us_accidents['Year'] = us_accidents['Start_Time'].dt.year
plt.figure(figsize=(10, 6))
us_accidents['Year'].value_counts().sort_index().plot(kind='bar')
plt.title('Accidents by Year')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.show()

print("Preprocessing the dataset...")
us_accidents = us_accidents.dropna(subset=['Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)'])
X = us_accidents[['Start_Lat', 'Start_Lng', 'Distance(mi)']].values
y = us_accidents['Severity'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y - 1, dtype=torch.long)  # Shift labels from 1-4 to 0-3

train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(list(zip(X_tensor, y_tensor)), [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AccidentSeverityModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

print("Evaluating the model...")
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_targets.extend(targets.numpy())

print("Accuracy:", accuracy_score(all_targets, all_preds))
print("Classification Report:")
print(classification_report(all_targets, all_preds))

print("Loading Traffic Flow dataset...")
traffic_flow_data = loadmat('traffic_dataset.mat')
print("Traffic Flow dataset loaded.")

print("Integrating traffic flow data...")
traffic_X_te = traffic_flow_data['tra_X_te'][0]
traffic_features = np.mean([x.mean(axis=1) for x in traffic_X_te], axis=0)
us_accidents['Traffic_Feature'] = np.random.choice(traffic_features.flatten(), size=us_accidents.shape[0], replace=True)
X = us_accidents[['Start_Lat', 'Start_Lng', 'Distance(mi)', 'Traffic_Feature']].values
y = us_accidents['Severity'].values -1

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(list(zip(X_tensor, y_tensor)), [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = AccidentSeverityModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Retraining the model with integrated data...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

print("Evaluating the integrated model...")
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_targets.extend(targets.numpy())

print("Integrated Model Accuracy:", accuracy_score(all_targets, all_preds))
print("Classification Report:")
print(classification_report(all_targets, all_preds))

print("Analysis complete.")
