import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

def stationarity_tests(series, company):
    print(f"\n--- Stationarity tests for {company} ---")
    adf_result = adfuller(series.dropna())
    pp_result = PhillipsPerron(series.dropna())
    kpss_result, _, _, _ = kpss(series.dropna(), nlags="auto")

    print(f"ADF p-value: {adf_result[1]:.4f}")
    print(f"PP p-value: {pp_result.pvalue:.4f}")
    print(f"KPSS statistic: {kpss_result:.4f}")

def plot_acf_pacf(series, company):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0])
    plot_pacf(series.dropna(), ax=axes[1])
    axes[0].set_title(f"ACF - {company}")
    axes[1].set_title(f"PACF - {company}")
    plt.tight_layout()
    plt.show()

def select_best_arima_order(series):
    best_aic = np.inf
    best_order = None
    for p in range(4):
        for q in range(4):
            try:
                model = ARIMA(series, order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
            except:
                continue
    return best_order

def test_arch_effects(series, lags=12):
    from statsmodels.stats.diagnostic import het_arch
    test_stat, p_value, _, _ = het_arch(series.dropna(), nlags=lags)
    return p_value < 0.05

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def forecast_arma_garch(df, company):
    series = df['Return'].dropna()

    # Skip constant series
    if series.nunique() <= 1:
        print(f"Skipping {company} — Return series is constant.")
        return

    try:
        stationarity_tests(series, company)
        plot_acf_pacf(series, company)
    except ValueError as e:
        print(f"Skipping {company} — Stationarity test failed: {e}")
        return

    order = select_best_arima_order(series)
    if not order or order == (0, 0):
        print(f"Skipping {company} — No valid ARMA order found.")
        return

    print(f"Selected ARMA order for {company}: {order}")

    is_arch = test_arch_effects(series)
    print(f"ARCH effect present: {is_arch}")

    train = series[:-84]
    test = series[-84:]

    forecast_values = None

    if is_arch:
        try:
            p, q = order
            am = arch_model(train, vol='Garch', p=1, q=1, mean='ARX', lags=p)
            res = am.fit(disp="off")
            forecast = res.forecast(horizon=84)
            forecast_values = forecast.mean.values[-1]
        except:
            print(f"Falling back to GARCH(1,1) for {company}")
            try:
                am = arch_model(train, vol='Garch', p=1, q=1)
                res = am.fit(disp="off")
                forecast = res.forecast(horizon=84)
                forecast_values = forecast.mean.values[-1]
            except:
                print(f"Skipping {company} — GARCH fallback also failed.")
                return
    else:
        try:
            model = ARIMA(train, order=(order[0], 0, order[1])).fit()
            forecast_values = model.forecast(steps=84)
        except:
            print(f"Skipping {company} — ARMA failed.")
            return

    forecast_values = np.ravel(forecast_values)
    forecast_values = forecast_values[:len(test)]

    mae, rmse, r2 = evaluate(test, forecast_values)
    print(f"{company} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test.values, label='Actual', marker='o')
    plt.plot(test.index, forecast_values, label='Forecast', marker='x')
    plt.title(f'{company} - Actual vs Forecasted Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_dir = "cleaned_data"
    companies = ['ALKM', 'AST', 'BHASS', 'PLTU']

    for company in companies:
        file_path = os.path.join(data_dir, f"cleaned_{company}.csv")
        if not os.path.exists(file_path):
            print(f"File not found for {company}")
            continue

        df = pd.read_csv(file_path)
        if 'Return' not in df.columns:
            print(f"'Return' column missing in {company}")
            continue

        forecast_arma_garch(df, company)
