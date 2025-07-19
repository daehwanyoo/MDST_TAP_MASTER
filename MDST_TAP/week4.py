import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv('filtered_data.csv')


# Ensure necessary columns exist
df = df[["Severity", "Start_Time", "End_Time", "Street", "Weather_Timestamp", "Crossing", "Traffic_Signal"]]

# Convert time columns to datetime format
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
df["Weather_Timestamp"] = pd.to_datetime(df["Weather_Timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
# Extract hour information from time-based features
df["Start_Hour"] = df["Start_Time"].dt.hour
df["End_Hour"] = df["End_Time"].dt.hour
df["Weather_Hour"] = df["Weather_Timestamp"].dt.hour

print(df["End_Hour"])
print(df[df["End_Time"].isna()])  
print(df["End_Time"].isna().sum())
print(df["End_Time"].dropna().astype(str).str.slice(0, 19).unique())

# # 1. Bar plot for severity count
# plt.figure(figsize=(8, 5))
# sns.countplot(x="Severity", data=df, palette="viridis")
# plt.title("Severity Count")
# plt.xlabel("Severity Level")
# plt.ylabel("Count")
# plt.show()

# # 2. Histogram of Start Hour
# plt.figure(figsize=(8, 5))
# sns.histplot(df["Start_Hour"], bins=24, kde=True)
# plt.title("Distribution of Start Times")
# plt.xlabel("Hour of the Day")
# plt.ylabel("Frequency")
# plt.show()

# 3. Histogram of End Hour
plt.figure(figsize=(8, 5))
sns.histplot(df["End_Hour"], bins=24, kde=True)
plt.title("Distribution of End Times")
plt.xlabel("Hour of the Day")
plt.ylabel("Frequency")
plt.show()

# 4. Boxplot of Severity vs. Start Hour
plt.figure(figsize=(8, 5))
sns.boxplot(x="Severity", y="Start_Hour", data=df)
plt.xlabel("Severity")
plt.ylabel("Start Hour")
plt.show()

# 5. Count plot for Traffic Signals in relation to Severity
plt.figure(figsize=(8, 5))
sns.countplot(x="Severity", hue=df["Traffic_Signal"], data=df, palette="coolwarm")
plt.title("Severity vs. Presence of Traffic Signal")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.legend(title="Traffic Signal")
plt.show()

# 6. Count plot for Crossings in relation to Severity
plt.figure(figsize=(8, 5))
sns.countplot(x="Severity", hue=df["Crossing"], data=df, palette="coolwarm")
plt.title("Severity vs. Presence of Crossing")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.legend(title="Crossing")
plt.show()

# 7. Pie chart for Severity distribution
plt.figure(figsize=(6, 6))
df["Severity"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Severity Distribution")
plt.ylabel("")
plt.show()

# 8. Violin plot of Severity vs. End Hour
plt.figure(figsize=(8, 5))
sns.violinplot(x="Severity", y="End_Hour", data=df, palette="muted")
plt.title("Severity vs. End Time")
plt.xlabel("Severity")
plt.ylabel("End Hour")
plt.show()

# 9. Heatmap of time correlations
plt.figure(figsize=(8, 5))
sns.heatmap(df[["Severity", "Start_Hour", "End_Hour", "Weather_Hour"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 10. Strip plot of Severity vs. Weather Hour
plt.figure(figsize=(8, 5))
sns.stripplot(x="Severity", y="Weather_Hour", data=df, jitter=True, palette="deep")
plt.title("Severity vs. Weather Report Time")
plt.xlabel("Severity")
plt.ylabel("Weather Report Hour")
plt.show()