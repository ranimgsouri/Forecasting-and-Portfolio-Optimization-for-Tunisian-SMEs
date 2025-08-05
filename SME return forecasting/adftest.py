from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# Load dataset
file_path = "cleaned_data/cleaned_CITY Historical Data.xlsx"
df = pd.read_excel(file_path)

# Data preparation
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
df.dropna(subset=['Log_Return'], inplace=True)

adf_result = adfuller(df['Log_Return'])
print("ADF Test:")
print(f"  Test Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")
print(f"  Critical Values: {adf_result[4]}")

# 2. Phillips-Perron
pp_result = PhillipsPerron(df['Log_Return'])
print("\nPhillips-Perron Test:")
print(f"  Test Statistic: {pp_result.stat:.4f}")
print(f"  p-value: {pp_result.pvalue:.4f}")

# 3. KPSS
kpss_result, kpss_pvalue, _, kpss_crit = kpss(df['Log_Return'], regression='c')
print("\nKPSS Test:")
print(f"  Test Statistic: {kpss_result:.4f}")
print(f"  p-value: {kpss_pvalue:.4f}")
print(f"  Critical Values: {kpss_crit}")
# Plot ACF/PACF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['Log_Return'], lags=20, ax=ax1)
plot_pacf(df['Log_Return'], lags=20, ax=ax2)
plt.show()

# ARMA model selection
models = {
    "ARMA(1,0)": (1, 0, 0),
    "ARMA(1,1)": (1, 0, 1),
    "ARMA(2,1)": (2, 0, 1),
    "ARMA(2,2)": (2, 0, 2),
    "ARMA(2,3)": (2, 0, 3),
    "ARMA(3,1)": (3, 0, 1),
    "Arma(3,2)": (3, 0, 2)
}

best_aic = float("inf")
best_arma_model = None

for name, order in models.items():
    try:
        model = ARIMA(df['Log_Return'], order=order).fit()
        print(f"{name} | AIC: {model.aic:.2f} | BIC: {model.bic:.2f}")
        if model.aic < best_aic:
            best_aic = model.aic
            best_arma_model = model
            best_order = order
    except Exception as e:
        print(f"Failed to fit {name}: {str(e)}")

print(f"\nBest ARMA Model: ARMA{best_order} (AIC: {best_aic:.2f})")

# ARCH effects test
def arch_test(residuals, lags=5):
    """Manual ARCH test implementation"""
    squared_resid = residuals**2
    lagged_values = pd.DataFrame(squared_resid)
    for i in range(1, lags+1):
        lagged_values[f'Lag_{i}'] = squared_resid.shift(i)
    lagged_values.dropna(inplace=True)
    y = lagged_values.iloc[:, 0]
    X = lagged_values.iloc[:, 1:]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.f_pvalue

arch_pvalue = arch_test(best_arma_model.resid)
print(f"\nARCH Test p-value: {arch_pvalue:.4f}")

# If ARCH effects detected, test volatility models
if arch_pvalue < 0.05:
    print("Significant ARCH effects detected - comparing GARCH family models")

    families = {
        'GARCH': {'vol':'GARCH'},
        'EGARCH': {'vol':'EGARCH'},
        'GJR-GARCH': {'vol':'GARCH', 'o':1}  # o=1 adds asymmetry
    }

    best_family = None
    best_family_aic = np.inf

    for name, specs in families.items():
        try:
            model = arch_model(best_arma_model.resid, p=1, q=1, **specs).fit(disp='off')
            print(f"{name}(1,1) | AIC: {model.aic:.2f}")
            if model.aic < best_family_aic:
                best_family = name
                best_family_aic = model.aic
                best_family_specs = specs
        except Exception as e:
            print(f"Failed to fit {name}: {e}")

    print(f"\nBest Volatility Model Family: {best_family}")

    # Now test different (p, q) parameters
    param_combinations = [
        (1,1), (1,2), (2,1), (2,2), (1,3), (3,1)
    ]

    best_vol_model = None
    best_vol_aic = np.inf

    print(f"\nTesting {best_family} Parameter Combinations:")
    for p, q in param_combinations:
        try:
            if best_family == 'GJR-GARCH':
                model = arch_model(best_arma_model.resid, p=p, o=1, q=q, **best_family_specs).fit(disp='off')
            else:
                model = arch_model(best_arma_model.resid, p=p, q=q, **best_family_specs).fit(disp='off')
            print(f"{best_family}({p},{q}) | AIC: {model.aic:.2f}")
            if model.aic < best_vol_aic:
                best_vol_aic = model.aic
                best_vol_model = model
                best_params = (p, q)
        except Exception as e:
            print(f"Failed to fit {best_family}({p},{q}): {e}")

    print(f"\n✅ Best Volatility Model: {best_family}{best_params} (AIC: {best_vol_aic:.2f})")
    print(best_vol_model.summary())
else:
    print("\nNo significant ARCH effects detected - volatility modeling not required")
# Forecasting with the best ARMA model
forecast_steps = 60  # Approx 2 months of trading days
forecast = best_arma_model.get_forecast(steps=forecast_steps)

# Get forecast and confidence intervals
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Create date index for forecast period
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')  # Business days

# Convert log returns back to price forecasts
last_price = df['Price'].iloc[-1]
price_forecast = [last_price * np.exp(forecast_mean.cumsum())]

# Plotting
plt.figure(figsize=(12, 6))

# Plot historical prices
plt.plot(df.index[-120:], df['Price'][-120:], label='Historical Prices', color='blue')

# Plot forecasted prices
plt.plot(forecast_dates, price_forecast[0], label='Forecasted Prices', color='red', linestyle='--')

# Plot confidence intervals (convert log return CI to price CI)
lower_bound = last_price * np.exp(forecast_conf_int.iloc[:, 0].cumsum())
upper_bound = last_price * np.exp(forecast_conf_int.iloc[:, 1].cumsum())
plt.fill_between(forecast_dates, lower_bound, upper_bound, color='pink', alpha=0.3, label='95% Confidence Interval')

plt.title(f'Price Forecast for Next {forecast_steps} Trading Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Create DataFrame with forecast results
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Log_Return_Forecast': forecast_mean.values,
    'Lower_CI': forecast_conf_int.iloc[:, 0].values,
    'Upper_CI': forecast_conf_int.iloc[:, 1].values,
    'Price_Forecast': price_forecast[0].values,
    'Price_Lower_CI': lower_bound.values,
    'Price_Upper_CI': upper_bound.values
}).set_index('Date')
# Split data into train and test (last 60 days for testing)
test_size = 60  
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

# Re-fit the best ARMA model on training data
best_order = (1, 0, 1)  # Replace with your best ARMA order from earlier
arma_model = ARIMA(train['Log_Return'], order=best_order).fit()

# Forecast log returns for the test period
forecast = arma_model.get_forecast(steps=test_size)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Convert log returns to price forecasts
last_train_price = train['Price'].iloc[-1]
cumulative_log_returns = forecast_mean.cumsum()
price_forecast = last_train_price * np.exp(cumulative_log_returns)

# Actual prices for the test period
actual_prices = test['Price']
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mae = mean_absolute_error(actual_prices, price_forecast)
rmse = np.sqrt(mean_squared_error(actual_prices, price_forecast))
mape = mean_absolute_percentage_error(actual_prices, price_forecast)

print("\nForecast Accuracy Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(train.index[-60:], train['Price'][-60:], label='Training Data', color='blue')
plt.plot(test.index, actual_prices, label='Actual Prices', color='green', linewidth=2)
plt.plot(test.index, price_forecast, label='Forecasted Prices', color='red', linestyle='--')
plt.fill_between(test.index,
                 last_train_price * np.exp(forecast_conf_int.iloc[:, 0].cumsum()),
                 last_train_price * np.exp(forecast_conf_int.iloc[:, 1].cumsum()),
                 color='pink', alpha=0.3, label='95% CI')
plt.title('Actual vs. Forecasted Prices (Test Period)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()