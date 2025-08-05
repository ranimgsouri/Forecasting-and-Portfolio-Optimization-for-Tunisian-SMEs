import os
import pandas as pd
from glob import glob

# Set your model results folder
model_results_dir = r"C:\Users\ranim\PFE\model_results"
output_file = r"C:\Users\ranim\PFE\forecasted_returns_matrix.csv"

# Get all CSV files only
csv_files = glob(os.path.join(model_results_dir, "*.csv"))

# Initialize list to collect data
forecasted_data = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        if 'ForecastedLogReturn' not in df.columns:
            print(f"Skipping {file}: 'ForecastedLogReturn' not found.")
            continue

        # Get company name from filename
        company_name = os.path.basename(file).replace(".csv", "")

        # Use date if available, else fallback to index
        if 'date' in df.columns:
            temp = df[['date', 'ForecastedLogReturn']].copy()
            temp['date'] = pd.to_datetime(temp['date'], errors='coerce')
            temp = temp.dropna(subset=['date'])
            temp = temp.set_index('date')
        else:
            temp = df[['ForecastedLogReturn']].copy()
            temp.index.name = 'date'

        temp = temp.rename(columns={'ForecastedLogReturn': company_name})
        forecasted_data.append(temp)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all into a single DataFrame
if not forecasted_data:
    raise ValueError("No valid forecast files found.")

forecast_matrix = pd.concat(forecasted_data, axis=1)

# Save to CSV
forecast_matrix.to_csv(output_file)

print("✅ Forecasted returns matrix saved to:", output_file)
print("Shape:", forecast_matrix.shape)
