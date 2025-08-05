import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# === GARCH family configuration ===
GARCH_CONFIG = {
    'AETEC': ('GJR-GARCH', (3, 1)),
    'BH': ('EGARCH', (3, 1)),
    'BHL': ('GARCH', (1, 1)),
    'DH': ('EGARCH', (1, 2)),
    'ECYCL': ('EGARCH', (3, 3)),
    'GIF': ('EGARCH', (3, 3)),
    'HANL': ('EGARCH', (2, 1)),
    'MIP': ('EGARCH', (1, 1)),
    'NBL': ('GARCH', (3, 3)),
    'PLAST': ('EGARCH', (2, 1)),
    'SAH': ('EGARCH', (2, 3)),
    'SERVI': ('EGARCH', (2, 3)),
    'SOPAT': ('EGARCH', (3, 1)),
    'STVR': ('EGARCH', (3, 3)),
    'TLNET': ('EGARCH', (3, 1))
}

# === Helper to get model ===
def get_vol_model(family, p, q):
    if family == 'GARCH':
        return arch_model(None, mean='ARX', lags=0, vol='GARCH', p=p, q=q, dist='skewt')
    elif family == 'EGARCH':
        return arch_model(None, mean='ARX', lags=0, vol='EGARCH', p=p, q=q, dist='skewt')
    elif family == 'GJR-GARCH':
        return arch_model(None, mean='ARX', lags=0, vol='GARCH', p=p, o=1, q=q, dist='skewt')
    else:
        raise ValueError("Unknown GARCH family")

# === Forecasting ===
forecast_horizon = 84  # 4 months (approx 21 days/month)
folder = 'cleaned_data'

for company, (family, (p, q)) in GARCH_CONFIG.items():
    print(f"🔮 Forecasting {company} | {family}({p},{q})")
    filename = f"cleaned_{company}.csv"
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        print(f"❌ Data not found for {company}")
        continue

    try:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        returns = df['Return'].dropna()

        model = get_vol_model(family, p, q)
        model = model.clone()
        model.y = returns

        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=forecast_horizon)
        mean_forecast = forecast.mean.iloc[-1].values
        variance_forecast = forecast.variance.iloc[-1].values

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(mean_forecast, label='Forecasted Return')
        plt.fill_between(range(len(mean_forecast)),
                         mean_forecast - 1.96 * np.sqrt(variance_forecast),
                         mean_forecast + 1.96 * np.sqrt(variance_forecast),
                         color='gray', alpha=0.3, label='95% CI')
        plt.title(f"{company} | {family}({p},{q}) Return Forecast (4 months)")
        plt.xlabel("Days")
        plt.ylabel("Forecasted Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"forecast_{company}.png")
        plt.close()

    except Exception as e:
        print(f"⚠️ Error processing {company}: {e}")
