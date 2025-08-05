import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# === Paths ===
DATA_FOLDER = 'cleaned_data'
RESULTS_SUMMARY_PATH = 'model_results/model_comparison_summary.csv'
RESULTS_FOLDER = os.path.join('results2', 'garch_forecasts')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

FORECAST_HORIZON = 80  # ~4 months

def forecast_garch(series, steps=80):
    try:
        model = arch_model(series, vol='GARCH', p=1, q=1)
        fitted_model = model.fit(disp='off')
        forecast = fitted_model.forecast(horizon=steps)
        return fitted_model, forecast.mean.iloc[-1].values  # model and forecasted means
    except Exception as e:
        print(f"GARCH error: {e}")
        return None, None

# Load model summary and get GARCH-best companies
summary_df = pd.read_csv(RESULTS_SUMMARY_PATH)
garch_companies = summary_df[summary_df['BestModel'] == 'GARCH']['Company'].tolist()

for company in garch_companies:
    print(f"\n📊 Processing {company}...")

    file_path = os.path.join(DATA_FOLDER, f'cleaned_{company}.csv')
    if not os.path.exists(file_path):
        print(f"Missing file for {company}. Skipping.")
        continue

    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date')

    if 'Price' not in df.columns:
        print(f"No 'Price' column for {company}. Skipping.")
        continue

    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff()
    series = df['LogReturn'].dropna()

    if len(series) < 100:
        print(f"Not enough data for {company}. Skipping.")
        continue

    model, forecasted_returns = forecast_garch(series, steps=FORECAST_HORIZON)
    if forecasted_returns is None:
        continue

    # === Save forecast CSV ===
    forecast_df = pd.DataFrame({
        'Day': range(1, FORECAST_HORIZON + 1),
        'ForecastedLogReturn': forecasted_returns
    })
    forecast_df.to_csv(os.path.join(RESULTS_FOLDER, f'{company}_garch_forecast.csv'), index=False)

    # === Plot 1: Forecasted Log Returns ===
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, FORECAST_HORIZON + 1), forecasted_returns, marker='o', linestyle='-')
    plt.title(f"{company} - GARCH Forecasted Log Returns")
    plt.xlabel("Day")
    plt.ylabel("Forecasted Log Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f'forecast_plot_{company}.png'))
    plt.close()

    # === Plot 2: Actual vs Fitted (in-sample) ===
plt.figure(figsize=(10, 4))
last_n = 200 if len(series) > 200 else len(series)
actual_vals = series[-last_n:].values
resid = model.resid[-last_n:]
mu = model.params.get('mu', 0)
fitted_vals = mu + resid

plt.plot(actual_vals, label='Actual', linewidth=2)
plt.plot(fitted_vals, label='Fitted (GARCH)', linestyle='--')
plt.title(f"{company} - Actual vs Fitted Log Returns")
plt.xlabel("Time")
plt.ylabel("Log Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f'actual_vs_forecast_plot_{company}.png'))
plt.close()


print("\n✅ All forecasts and plots saved to:", RESULTS_FOLDER)
