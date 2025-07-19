import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

print("Loading dataset...")
df = pd.read_csv("data/us_accidents_data_cleaned.csv").sample(frac=0.01, random_state=42)
print(f"Sample size: {len(df)}")

# Preserve original labels for evaluation
true_labels = df['Severity'].values.copy()

# Normalize
for col in ['Accident_Duration', 'Distance(mi)', 'Humidity(%)', 'Visibility(mi)']:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# One-hot encode
encoded = pd.get_dummies(df[['Weather_Condition', 'Sunrise_Sunset']], drop_first=False)
df = pd.concat([df, encoded], axis=1)

# Get all possible columns from training
feature_columns_14 = ['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag',
                      'Distance(mi)', 'Start_Hour_Sin', 'Start_Hour_Cos',
                      'Start_Month_Sin', 'Start_Month_Cos', 'Accident_Duration',
                      'Humidity(%)', 'Visibility(mi)'] + list(encoded.columns)

feature_columns_23 = feature_columns_14  # same structure in both training scripts

# Ensure all one-hot columns exist
for col in set(feature_columns_14 + feature_columns_23):
    if col not in df.columns:
        df[col] = 0
df = df.sort_index(axis=1)  # ensure consistent column order

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

# Load both models
model_14 = BinaryClassifier(len(feature_columns_14))
model_14.load_state_dict(torch.load("severity_1_vs_4_model.pth"))
model_14.eval()

model_23 = BinaryClassifier(len(feature_columns_23))
model_23.load_state_dict(torch.load("severity_2_vs_3_model.pth"))
model_23.eval()

# Predictions
predictions = []
with torch.no_grad():
    for i in range(len(df)):
        sample = df.iloc[i]
        severity = true_labels[i]

        if severity in [1, 4]:
            x = torch.tensor(sample[feature_columns_14].values.astype(np.float32)).unsqueeze(0)
            pred = model_14(x)
            mapped = torch.argmax(pred, dim=1).item()
            predictions.append(1 if mapped == 0 else 4)

        elif severity in [2, 3]:
            x = torch.tensor(sample[feature_columns_23].values.astype(np.float32)).unsqueeze(0)
            pred = model_23(x)
            mapped = torch.argmax(pred, dim=1).item()
            predictions.append(2 if mapped == 0 else 3)

        else:
            predictions.append(severity)

print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=[
    "Severity 1", "Severity 2", "Severity 3", "Severity 4"
]))
