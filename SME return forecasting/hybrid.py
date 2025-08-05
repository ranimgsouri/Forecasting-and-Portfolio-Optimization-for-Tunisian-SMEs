import os
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model  # GARCH
warnings.filterwarnings("ignore")

# === Config ===
DATA_FOLDER = 'cleaned_data'
RESULTS_FOLDER = 'results2'
FORECAST_HORIZON = 80 
LAGS = 5 
TEST_SIZE = 0.2  
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def evaluate_model(true, pred):
    return {
        'MAE': mean_absolute_error(true, pred),
        'RMSE': np.sqrt(mean_squared_error(true, pred))
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
    return model, y_train, y_test, y_pred

def fit_garch_model(series):
    garch_model = arch_model(series, vol='Garch', p=1, q=1)  # You can adjust p and q
    garch_fit = garch_model.fit(disp="off")
    return garch_fit

def hybrid_forecast_xgboost_garch(model, garch_model, last_values, lags=LAGS, steps=FORECAST_HORIZON):
    preds = []
    values = list(last_values)

    for _ in range(steps):
        X_input = np.array(values[-lags:]).reshape(1, -1)
        pred_return = model.predict(X_input)[0]
        
        # Forecasting volatility
        garch_forecast = garch_model.forecast(horizon=1)
        forecasted_volatility = np.sqrt(garch_forecast.variance.values[-1, :][0])  # Get forecasted volatility
        
        # Combine point prediction and the forecasted volatility
        preds.append(pred_return)
        values.append(pred_return)

    return preds, np.array(preds) * forecasted_volatility

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
    model, y_train, y_test, y_pred = fit_xgboost_train_test(series)

    # === Fit GARCH model ===
    garch_fit = fit_garch_model(series)

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
        'RMSE': metrics['RMSE']
    })
    print(f"✅ MAE: {metrics['MAE']:.6f} | RMSE: {metrics['RMSE']:.6f}")

    # === Forecast Future using Hybrid Model ===
    last_lags = series[-LAGS:]
    future_preds, future_volatility = hybrid_forecast_xgboost_garch(model, garch_fit, last_lags)
    
    # Save forecast
    forecast_df = pd.DataFrame({
        'Day': range(1, FORECAST_HORIZON + 1),
        'ForecastedLogReturn': future_preds,
        'ForecastedVolatility': future_volatility
    })
    forecast_df.to_csv(os.path.join(RESULTS_FOLDER, f'{company}_hybrid_forecast.csv'), index=False)

    # === Forecast Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(forecast_df['Day'], forecast_df['ForecastedLogReturn'], label='Forecasted Log Returns', color='orange')
    plt.title(f"{company} - Forecast (Next {FORECAST_HORIZON} Days)")
    plt.xlabel("Day Ahead")
    plt.ylabel("Forecasted Log Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f'{company}_hybrid_plot.png'))
    plt.close()

# Save evaluation summary
pd.DataFrame(all_metrics).to_csv(os.path.join(RESULTS_FOLDER, 'hybrid_evaluation_summary.csv'), index=False)
print("\n📁 All results and plots saved in 'results2/' folder.")
