import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("filtered_data.csv")

# Handle Start_Time formatting issue
df['Start_Time'] = df['Start_Time'].astype(str).str.split('.').str[0]  # Remove extra precision if exists
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')  # Convert to datetime

# Extract relevant time-based features
df['Hour'] = df['Start_Time'].dt.hour
df['Month'] = df['Start_Time'].dt.month

# Fix missing values for numerical columns
df['Wind_Speed(mph)'] = pd.to_numeric(df['Wind_Speed(mph)'], errors='coerce').fillna(0)
df['Precipitation(in)'] = pd.to_numeric(df['Precipitation(in)'], errors='coerce').fillna(0)

# Ensure Severity is numeric
df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')

# Pie Charts
severity_counts = df['Severity'].value_counts(normalize=True) * 100
plt.figure(figsize=(6, 6))
plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Traffic Condition: Severity Distribution")
plt.show()

# Road conditions presence
road_conditions = ['Crossing', 'Traffic_Signal', 'Junction']
for condition in road_conditions:
    counts = df[condition].astype(bool).value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=['Absent', 'Present'], autopct='%1.1f%%', startangle=140)
    plt.title(f"Presence of {condition} (True/False)")
    plt.show()

# Bar Plots
# Accident Cases vs Hours
hourly_counts = df['Hour'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(hourly_counts.index, hourly_counts.values)
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases vs Hours")
plt.show()

# Accident Cases vs Months
monthly_counts = df['Month'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_counts.index, monthly_counts.values)
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases vs Months")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Accident Cases vs Different Temperature
plt.figure(figsize=(10, 5))
plt.hist(df['Temperature(F)'].dropna(), bins=30, edgecolor='black')
plt.xlabel("Temperature (F)")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases vs Different Temperature")
plt.show()

# Accident Cases vs Different Humidity
plt.figure(figsize=(10, 5))
plt.hist(df['Humidity(%)'].dropna(), bins=30, edgecolor='black')
plt.xlabel("Humidity (%)")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases vs Different Humidity")
plt.show()

# Accident Cases vs Wind Speed
plt.figure(figsize=(10, 5))
plt.hist(df['Wind_Speed(mph)'], bins=30, edgecolor='black')
plt.xlabel("Wind Speed (mph)")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases vs Wind Speed")
plt.show()
