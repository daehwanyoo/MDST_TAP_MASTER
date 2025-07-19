import pandas as pd
import scipy.stats as stats

# Load dataset
df = pd.read_csv("US_Accidents_March23.csv")

# Select numerical and categorical columns
columns_to_test = ['Severity', 'Source', 'Start_Time', 'End_Time', 'Description', 'Street', 
                   'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone', 
                   'Airport_Code', 'Weather_Timestamp', 'Wind_Direction', 'Weather_Condition', 
                   'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 
                   'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 
                   'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 
                   'Astronomical_Twilight']

# Convert categorical variables to numerical using ordinal encoding
df_encoded = df[columns_to_test].copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes  # Convert categories to numeric codes

# Compute Spearman correlation
spearman_corr = df_encoded.corr(method='spearman')['Severity'].drop('Severity')

# Categorize correlation strength
def categorize_correlation(r):
    if 0.0 < abs(r) < 0.1:
        return "No correlation"
    elif 0.1 <= abs(r) < 0.3:
        return "Low correlation"
    elif 0.3 <= abs(r) < 0.5:
        return "Medium correlation"
    elif 0.5 <= abs(r) < 0.7:
        return "High correlation"
    elif 0.7 <= abs(r) <= 1:
        return "Very high correlation"
    else:
        return "Invalid"

# Create results DataFrame
correlation_results = pd.DataFrame({
    'Feature': spearman_corr.index,
    'Spearman Correlation': spearman_corr.values,
    'Strength': spearman_corr.apply(categorize_correlation).values
})

# Print results to console
print(correlation_results.to_string(index=False))

# Save to CSV
correlation_results.to_csv("spearman_correlation_analysis_1.csv", index=False)
print("\nSpearman correlation analysis saved as 'spearman_correlation_analysis.csv'")
