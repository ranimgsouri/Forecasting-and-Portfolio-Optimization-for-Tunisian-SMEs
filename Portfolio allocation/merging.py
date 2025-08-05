import os
import pandas as pd
from glob import glob

cleaned_dir = r"C:\Users\ranim\PFE\Cleaned"
output_file = r"C:\Users\ranim\PFE\returns_matrix.csv"

# Get all cleaned CSV files
csv_files = glob(os.path.join(cleaned_dir, "*.csv"))

# Create an empty list to store DataFrames
dfs = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Ensure required columns exist
        if not all(col in df.columns for col in ['date', 'return']):
            print(f"Skipping {file}: missing required columns")
            continue
            
        company_name = os.path.basename(file).replace(".csv", "")
        df['company'] = company_name
        # Convert date column explicitly
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Drop rows with invalid dates
        dfs.append(df[['date', 'return', 'company']])
    except Exception as e:
        print(f"Error processing {file}: {e}")

if not dfs:
    raise ValueError("No valid data found in any files")

# Combine all DataFrames
combined = pd.concat(dfs)

# Pivot to create returns matrix
returns_matrix = combined.pivot(index='date', columns='company', values='return')

# Sort by date and ensure numeric returns
returns_matrix = returns_matrix.sort_index()
returns_matrix = returns_matrix.apply(pd.to_numeric, errors='coerce')

# Save the merged returns matrix
returns_matrix.to_csv(output_file)

print("\n=== Final Returns Matrix Info ===")
print(f"Saved to: {output_file}")
print(f"Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
print(f"Shape: {returns_matrix.shape}")
print(f"Number of companies: {len(returns_matrix.columns)}")