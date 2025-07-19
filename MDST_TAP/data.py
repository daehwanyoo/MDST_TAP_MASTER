import pandas as pd

# Load dataset
df = pd.read_csv("US_Accidents_March23.csv")

# Extract every 77th row
filtered_df = df.iloc[::77]

# Save to a new CSV file
filtered_df.to_csv("filtered_data.csv", index=False)

print("Filtered dataset saved as 'filtered_data.csv'")
