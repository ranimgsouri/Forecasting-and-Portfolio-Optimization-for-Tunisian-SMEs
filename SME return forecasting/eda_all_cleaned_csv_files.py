import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from scipy.stats import skew
import os

# Set folders
input_dir = "cleaned_data"
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

# Loop through all cleaned CSV files
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        company_name = file.replace("cleaned_", "").replace(".csv", "")
        print(f"\n📊 EDA for: {company_name}")

        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        df['Return'] = df['Price'].pct_change() * 100
        df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
        df['Simple_Return'] = df['Price'].pct_change()

        # Print summaries
        print("First rows:")
        print(df.head())
        print("\nSummary:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())

        # Skewness
        skew_log = skew(df['Log_Return'].dropna())
        skew_simple = skew(df['Simple_Return'].dropna())
        print(f"Skewness (Log Returns): {skew_log:.3f}")
        print(f"Skewness (Simple Returns): {skew_simple:.3f}")

        # Plot
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(df.index, df["Price"], label="Stock Price", color="blue")
        plt.title(f"{company_name} Price Over Time")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(df.index, df["Return"], label="Returns", color="green")
        plt.title(f"{company_name} Returns Over Time")
        plt.grid()

        plt.subplot(2, 2, 3)
        sns.histplot(df["Return"].dropna(), kde=True, color="purple")
        plt.title("Return Distribution")

        plt.subplot(2, 2, 4)
        sns.boxplot(x=df["Return"].dropna(), color="orange")
        plt.title("Return Boxplot")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{company_name}_EDA.png"))
        plt.close()

        # Q-Q Plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        qqplot(df['Log_Return'].dropna(), line='s', fit=True)
        plt.title("Q-Q Plot: Log Returns")

        plt.subplot(1, 2, 2)
        qqplot(df['Simple_Return'].dropna(), line='s', fit=True)
        plt.title("Q-Q Plot: Simple Returns")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{company_name}_QQ.png"))
        plt.close()

        print(f"📈 Plots saved for {company_name}")
