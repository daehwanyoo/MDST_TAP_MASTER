import scipy.stats as stats

import pandas as pd

# Load your preprocessed dataset
df = pd.read_csv("US_Accidents_March23.csv")

# Select numerical columns from your dataset
numerical_cols = ['Severity', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 
                  'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
                  'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

# Compute Pearson correlation coefficients and p-values
correlation_results = []
for col in numerical_cols:
    if col != "Severity":  # Exclude self-correlation
        # Drop rows where either 'Severity' or the selected column is NaN
        subset = df[['Severity', col]].dropna()
        
        # Compute correlation only if enough valid data points exist
        if len(subset) > 1:  # Pearson correlation requires at least 2 values
            r, p_value = stats.pearsonr(subset["Severity"], subset[col])
        else:
            r, p_value = None, None  # Not enough data to compute correlation
        
        # Categorize correlation strength
        if r is None:
            strength = "Not enough data"
        elif 0.0 < abs(r) < 0.1:
            strength = "No correlation"
        elif 0.1 <= abs(r) < 0.3:
            strength = "Low correlation"
        elif 0.3 <= abs(r) < 0.5:
            strength = "Medium correlation"
        elif 0.5 <= abs(r) < 0.7:
            strength = "High correlation"
        elif 0.7 <= abs(r) <= 1:
            strength = "Very high correlation"
        else:
            strength = "Invalid"

        # Append to results
        correlation_results.append({
            "Feature": col,
            "Correlation": r,
            "P-value": p_value,
            "Strength": strength
        })

# Convert results to DataFrame
correlation_df = pd.DataFrame(correlation_results)

# Print results to console
print(correlation_df.to_string(index=False))

# Save results to CSV for further analysis
correlation_df.to_csv("correlation_analysis_2.csv", index=False)
print("\nCorrelation analysis saved as 'correlation_analysis.csv'")