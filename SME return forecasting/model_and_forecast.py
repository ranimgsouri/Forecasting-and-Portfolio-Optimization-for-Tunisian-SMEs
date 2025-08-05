import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Paths
data_folder = 'cleaned_data'
forecast_horizon = 10  # Forecast 10 periods ahead
results_folder = 'forecast_results'
os.makedirs(results_folder, exist_ok=True)

# Loop through all CSVs
for file in os.listdir(data_folder):
    if not file.endswith(".csv"):
        continue

    company_name = file.replace("cleaned_", "").replace(".csv", "")
    print(f"\n=== Processing: {file} ===")

    # Load and preprocess
    df = pd.read_csv(os.path.join(data_folder, file), parse_dates=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # Ensure column exists
    if 'Price' not in df.columns:
        print("❌ No 'Price' column found. Skipping.")
        continue

    # Drop NA and take log returns
    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff().dropna()
    series = df['LogReturn'].dropna()

    # ADF Test
    adf_result = adfuller(series)
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    for key, value in adf_result[4].items():
        print(f"Critical Value ({key}%): {value}")
    
    d = 0
    if adf_result[1] > 0.05:
        print("❌ Series is NOT stationary, differencing will be applied")
        series = series.diff().dropna()
        d = 1
    else:
        print("✅ Series is stationary")

    # Fit ARIMA
    best_order = (0, d, 0)
    best_aic = float('inf')
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
            except:
                continue

    print(f"Best ARIMA model: {best_order}")

    # Fit best model
    model = ARIMA(series, order=best_order)
    model_fit = model.fit()

    # Ljung-Box test
    ljung_p = acorr_ljungbox(model_fit.resid, lags=[10], return_df=True).iloc[0, 1]
    print(f"Ljung-Box test p-value: {ljung_p}")
    if ljung_p > 0.05:
        print("✅ Residuals are uncorrelated")
    else:
        print("❌ Residuals are correlated")

    # Forecast
    forecast = model_fit.forecast(steps=forecast_horizon)
    forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

    # Align with actual data (if available)
    if len(series) >= forecast_horizon:
        actual = series[-forecast_horizon:]
        aligned_df = pd.DataFrame({
            'Actual': actual.values,
            'Forecast': forecast[:len(actual)].values
        })
        if not aligned_df.empty and len(aligned_df.dropna()) > 0:
            mae = mean_absolute_error(aligned_df['Actual'], aligned_df['Forecast'])
            print(f"MAE: {mae:.6f}")
        else:
            print("⚠️ Not enough overlapping data to compute MAE.")
    else:
        print("⚠️ Not enough actual data to compare forecasts.")

    # Plot forecast vs original
    plt.figure(figsize=(10, 4))
    plt.plot(series[-50:], label='Historical')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
    plt.title(f"{company_name} Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"{company_name}_forecast.png"))
    plt.close()
