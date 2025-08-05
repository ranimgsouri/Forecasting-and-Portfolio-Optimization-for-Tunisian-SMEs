import os
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore")

# Paths
DATA_FOLDER = 'cleaned_data'
RESULTS_FOLDER = 'model_results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Utility functions
def evaluate_model(true, pred):
    return {
        'MAE': mean_absolute_error(true, pred),
        'RMSE': np.sqrt(mean_squared_error(true, pred))
    }

def fit_arma(series):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(series, order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
                    best_model = model
            except:
                continue
    return best_model

def fit_expsmooth(series):
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
        return model
    except:
        return None

def fit_xgboost(series):
    try:
        df = pd.DataFrame({'y': series})
        for lag in range(1, 6):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        df = df.dropna()

        X = df.drop(columns=['y']).values
        y = df['y'].values

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        preds = model.predict(X)
        return model, y, preds
    except:
        return None, None, None

def fit_lstm(series):
    try:
        data = series.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(5, len(scaled_data)):
            X.append(scaled_data[i-5:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        model.fit(X, y, epochs=20, verbose=0)

        preds = model.predict(X)
        preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        y_inv = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

        return model, y_inv, preds_inv
    except:
        return None, None, None

# Main loop
results = []

for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".csv"):
        continue

    company = file.replace("cleaned_", "").replace(".csv", "")
    print(f"\n=== {company} ===")
    df = pd.read_csv(os.path.join(DATA_FOLDER, file), parse_dates=['Date'])
    df = df.sort_values('Date')

    if 'Price' not in df.columns:
        print("No 'Price' column. Skipping.")
        continue

    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff()
    series = df['LogReturn'].dropna()

    if len(series) < 100:
        print("Not enough data. Skipping.")
        continue

    best_model_name = None
    best_metrics = {'RMSE': np.inf}

    # ARMA
    arma_model = fit_arma(series)
    if arma_model:
        arma_pred = arma_model.fittedvalues
        arma_metrics = evaluate_model(series[arma_pred.index], arma_pred)
        if arma_metrics['RMSE'] < best_metrics['RMSE']:
            best_model_name, best_metrics = 'ARMA', arma_metrics

    # Exponential Smoothing
    exp_model = fit_expsmooth(series)
    if exp_model:
        exp_pred = exp_model.fittedvalues
        exp_metrics = evaluate_model(series[exp_pred.index], exp_pred)
        if exp_metrics['RMSE'] < best_metrics['RMSE']:
            best_model_name, best_metrics = 'ExponentialSmoothing', exp_metrics

    # XGBoost
    xgb_model, xgb_y, xgb_pred = fit_xgboost(series)
    if xgb_model:
        xgb_metrics = evaluate_model(xgb_y, xgb_pred)
        if xgb_metrics['RMSE'] < best_metrics['RMSE']:
            best_model_name, best_metrics = 'XGBoost', xgb_metrics

    # LSTM
    lstm_model, lstm_y, lstm_pred = fit_lstm(series)
    if lstm_model:
        lstm_metrics = evaluate_model(lstm_y, lstm_pred)
        if lstm_metrics['RMSE'] < best_metrics['RMSE']:
            best_model_name, best_metrics = 'LSTM', lstm_metrics

    print(f"Best model: {best_model_name}")
    results.append({
        'Company': company,
        'BestModel': best_model_name,
        'MAE': best_metrics['MAE'],
        'RMSE': best_metrics['RMSE']
    })

# Save all results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_FOLDER, 'model_comparison_summary.csv'), index=False)
print("\n✅ Model comparison complete. Summary saved.")

FORECAST_HORIZON = 80  # About 4 months of trading days

def forecast_xgboost(series, model, lags=5, steps=80):
    last_values = list(series[-lags:])
    preds = []

    for _ in range(steps):
        X_input = np.array(last_values[-lags:]).reshape(1, -1)
        pred = model.predict(X_input)[0]
        preds.append(pred)
        last_values.append(pred)  # recursive: use prediction as new input

    return preds

forecast_results = []

for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".csv"):
        continue

    company = file.replace("cleaned_", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(DATA_FOLDER, file), parse_dates=['Date'])
    df = df.sort_values('Date')

    if 'Price' not in df.columns:
        continue

    df['LogPrice'] = np.log(df['Price'].replace(0, np.nan))
    df['LogReturn'] = df['LogPrice'].diff()
    series = df['LogReturn'].dropna()

    if len(series) < 100:
        continue

    # === XGBoost Refit (since best model previously selected as XGBoost)
    xgb_model, xgb_y, xgb_pred = fit_xgboost(series)

    if xgb_model:
        future_preds = forecast_xgboost(series, xgb_model, lags=5, steps=FORECAST_HORIZON)

        # Save forecast to CSV
        forecast_df = pd.DataFrame({
            'Day': range(1, FORECAST_HORIZON + 1),
            'ForecastedLogReturn': future_preds
        })
        forecast_df.to_csv(os.path.join(RESULTS_FOLDER, f'{company}_xgboost_forecast.csv'), index=False)

        forecast_results.append({'Company': company, 'ForecastDays': FORECAST_HORIZON})

print("\n📈 Forecasting complete for selected XGBoost models. Forecasts saved.")
