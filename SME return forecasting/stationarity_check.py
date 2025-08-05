import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

# Paths
data_folder = 'cleaned_data'
stationarity_results_folder = 'stationarity_results'
os.makedirs(stationarity_results_folder, exist_ok=True)

def run_stationarity_tests(series, company_name):
    results = {}

    # ADF Test
    adf_stat, adf_p, _, _, adf_crit_vals, _ = adfuller(series)
    results['ADF'] = {'Statistic': adf_stat, 'p-value': adf_p, 'Critical Values': adf_crit_vals}

    # KPSS Test
    kpss_stat, kpss_p, _, kpss_crit_vals = kpss(series, regression='c', nlags="legacy")
    results['KPSS'] = {'Statistic': kpss_stat, 'p-value': kpss_p, 'Critical Values': kpss_crit_vals}

    # Phillips-Perron Test
    pp_test = PhillipsPerron(series)
    results['PP'] = {'Statistic': pp_test.stat, 'p-value': pp_test.pvalue, 'Critical Values': pp_test.critical_values}

    # Save results to text file
    with open(os.path.join(stationarity_results_folder, f"{company_name}_stationarity.txt"), 'w') as f:
        for test_name, res in results.items():
            f.write(f"=== {test_name} Test ===\n")
            for key, val in res.items():
                f.write(f"{key}: {val}\n")
            f.write("\n")

    # Plot ACF and PACF
    # Determine safe number of lags based on sample size
    max_lags = int(len(series) * 0.5)
    lags_to_use = min(40, max_lags)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, ax=axes[0], lags=lags_to_use)
    plot_pacf(series, ax=axes[1], lags=lags_to_use)

    axes[0].set_title(f'{company_name} - ACF')
    axes[1].set_title(f'{company_name} - PACF')
    plt.tight_layout()
    plt.savefig(os.path.join(stationarity_results_folder, f"{company_name}_acf_pacf.png"))
    plt.close()

# Loop through all CSVs
for file in os.listdir(data_folder):
    if not file.endswith(".csv"):
        continue

    company_name = file.replace("cleaned_", "").replace(".csv", "")
    print(f"\n=== Stationarity Check: {file} ===")

    # Load and preprocess
    df = pd.read_csv(os.path.join(data_folder, file), parse_dates=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    if 'Price' not in df.columns:
        print("❌ No 'Price' column found. Skipping.")
        continue

    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff().dropna()
    series = df['LogReturn'].dropna()

    if series.empty:
        print("❌ Log return series is empty. Skipping.")
        continue

    run_stationarity_tests(series, company_name)
