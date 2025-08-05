import pandas as pd
import os

# Paths
input_dir = "data"
output_dir = "cleaned_data"
os.makedirs(output_dir, exist_ok=True)

# Volume cleaning function
def convert_volume(value):
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if "K" in value:
            return float(value.replace("K", "")) * 1_000
        elif "M" in value:
            return float(value.replace("M", "")) * 1_000_000
    try:
        return float(value)
    except ValueError:
        return None

# Loop through all CSV files in the data folder
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        try:
            company_name = file.replace(" Historical Data.csv", "")
            print(f"📂 Processing: {company_name}")

            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"cleaned_{company_name}.csv")

            # Load CSV
            df = pd.read_csv(input_path)

            # Convert Date
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Convert numeric columns
            numeric_cols = ["Price", "Open", "High", "Low", "Change %"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce")

            # Convert Volume
            df["Vol."] = df["Vol."].apply(convert_volume)

            # Drop missing and sort
            df.dropna(inplace=True)
            df.sort_values("Date", inplace=True)

            # Calculate returns
            df['Return'] = df['Price'].pct_change() * 100

            # Remove return outliers using IQR
            Q1 = df['Return'].quantile(0.25)
            Q3 = df['Return'].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df['Return'] >= Q1 - 1.5 * IQR) & (df['Return'] <= Q3 + 1.5 * IQR)]

            # Save cleaned version
            df.to_csv(output_path, index=False)
            print(f"✅ Saved cleaned CSV: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
