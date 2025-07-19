import pandas as pd

# Load your dataset
file_path = 'data/us_accidents_data_cleaned.csv'
us_accidents = pd.read_csv(file_path)

# --- Distance(mi) ---
distance_mean = us_accidents['Distance(mi)'].mean()
distance_std = us_accidents['Distance(mi)'].std()
us_accidents['Distance(mi)'] = (us_accidents['Distance(mi)'] - distance_mean) / distance_std
distance_min = us_accidents['Distance(mi)'].min()
distance_max = us_accidents['Distance(mi)'].max()

# --- Accident_Duration ---
duration_mean = us_accidents['Accident_Duration'].mean()
duration_std = us_accidents['Accident_Duration'].std()
us_accidents['Accident_Duration'] = (us_accidents['Accident_Duration'] - duration_mean) / duration_std
duration_min = us_accidents['Accident_Duration'].min()
duration_max = us_accidents['Accident_Duration'].max()

print(f"Standardized Distance(mi) range: [{distance_min:.2f}, {distance_max:.2f}]")
print(f"Standardized Accident_Duration range: [{duration_min:.2f}, {duration_max:.2f}]")
