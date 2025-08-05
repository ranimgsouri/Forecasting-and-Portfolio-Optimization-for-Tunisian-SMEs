import os
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Paths
data_folder = 'cleaned_data'
decomposition_folder = 'decomposition_results'
os.makedirs(decomposition_folder, exist_ok=True)

# Loop through all CSVs
for file in os.listdir(data_folder):
    if not file.endswith(".csv"):
        continue

    company_name = file.replace("cleaned_", "").replace(".csv", "")
    print(f"🔍 Decomposing: {company_name}")

    # Load and preprocess
    df = pd.read_csv(os.path.join(data_folder, file), parse_dates=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    if 'Price' not in df.columns:
        print("❌ No 'Price' column found. Skipping.")
        continue

    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff()

    series = df['LogReturn'].dropna()

    if series.empty or len(series) < 30:
        print("⚠️ Not enough data for decomposition. Skipping.")
        continue

    try:
        # Apply additive decomposition (removes trend, returns residual)
        decomposition = seasonal_decompose(series, model='additive', period=30, extrapolate_trend='freq')

        # Plot decomposition
        plt.figure(figsize=(12, 6))
        plt.subplot(311)
        plt.plot(decomposition.observed, label='Observed', color='blue')
        plt.title(f'{company_name} - Log Return Series')

        plt.subplot(312)
        plt.plot(decomposition.trend, label='Trend', color='green')
        plt.title('Trend Component')

        plt.subplot(313)
        plt.plot(decomposition.resid, label='Residual', color='red')
        plt.title('Residual Component')

        plt.tight_layout()
        plt.savefig(os.path.join(decomposition_folder, f"{company_name}_decomposition.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ Decomposition failed for {company_name}: {e}")
