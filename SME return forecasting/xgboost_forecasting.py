import os
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# === Config ===
DATA_FOLDER = 'cleaned_data'
RESULTS_FOLDER = 'model_results'
FORECAST_HORIZON = 80
LAGS = 5
TEST_SIZE = 0.2  # 20% for testing
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    mean_actual = np.mean(np.abs(true))
    accuracy = 1 - (mae / mean_actual) if mean_actual != 0 else 0
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': accuracy
    }

def fit_xgboost_train_test(series, lags=LAGS, test_size=TEST_SIZE):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df = df.dropna()

    X = df.drop(columns=['y']).values
    y = df['y'].values

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, y_train, y_test, y_pred, df

def forecast_xgboost(model, last_values, lags=LAGS, steps=FORECAST_HORIZON):
    preds = []
    values = list(last_values)

    for _ in range(steps):
        X_input = np.array(values[-lags:]).reshape(1, -1)
        pred = model.predict(X_input)[0]
        preds.append(pred)
        values.append(pred)

    return preds

# === Main Loop ===
all_metrics = []

for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".csv"):
        continue

    company = file.replace("cleaned_", "").replace(".csv", "")
    print(f"\n📊 {company}")

    df = pd.read_csv(os.path.join(DATA_FOLDER, file), parse_dates=['Date'])
    df = df.sort_values('Date')

    if 'Price' not in df.columns:
        print("Missing 'Price'. Skipping.")
        continue

    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff()
    series = df['LogReturn'].dropna()

    if len(series) < 100:
        print("Too little data. Skipping.")
        continue

    # === Train/test fit ===
    model, y_train, y_test, y_pred, df_lagged = fit_xgboost_train_test(series)

    # === Accuracy Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(y_test)), y_test, label='Actual', linewidth=2)
    plt.plot(range(len(y_test)), y_pred, label='Predicted', linewidth=2)
    plt.title(f"{company} - XGBoost Test Accuracy")
    plt.xlabel("Time Index")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{company}_xgboost_accuracy.png'))
    plt.close()

    # === Accuracy Metrics ===
    metrics = evaluate_model(y_test, y_pred)
    all_metrics.append({
        'Company': company,
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
        'Accuracy': metrics['Accuracy']
    })
    print(f"✅ MAE: {metrics['MAE']:.6f} | RMSE: {metrics['RMSE']:.6f} | R²: {metrics['R2']:.4f} | Accuracy: {metrics['Accuracy']:.2%}")

    # === Forecast Future ===
    last_lags = series[-LAGS:]
    future_preds = forecast_xgboost(model, last_lags)

    # Save forecast
    forecast_df = pd.DataFrame({
        'Day': range(1, FORECAST_HORIZON + 1),
        'ForecastedLogReturn': future_preds
    })
    forecast_df.to_csv(os.path.join(RESULTS_FOLDER, f'{company}_xgboost_forecast.csv'), index=False)

    # === Forecast Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(forecast_df['Day'], forecast_df['ForecastedLogReturn'], label='Forecasted Log Returns', color='orange')
    plt.title(f"{company} - Forecast (Next {FORECAST_HORIZON} Days)")
    plt.xlabel("Day Ahead")
    plt.ylabel("Forecasted Log Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{company}_xgboost_plot.png'))
    plt.close()

# Save evaluation summary
pd.DataFrame(all_metrics).to_csv(os.path.join(RESULTS_FOLDER, 'xgboost_evaluation_summary.csv'), index=False)
print("\n📁 All results and plots saved in 'model_results/' folder.")
