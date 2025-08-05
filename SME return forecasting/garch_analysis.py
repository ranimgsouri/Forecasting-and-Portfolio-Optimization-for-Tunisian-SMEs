import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

DATA_FOLDER = 'cleaned_data'
OUTPUT_FOLDER = 'garch_results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def test_arch_effects(residuals):
    test_result = het_arch(residuals)
    p_value = test_result[1]
    return p_value

def compare_vol_models(series):
    models = {
        'GARCH': lambda: arch_model(series, vol='GARCH', p=1, q=1, mean='Zero'),
        'EGARCH': lambda: arch_model(series, vol='EGARCH', p=1, q=1, mean='Zero'),
        'GJR-GARCH': lambda: arch_model(series, vol='GARCH', p=1, q=1, o=1, mean='Zero')
    }

    best_model = None
    best_name = None
    best_aic = np.inf

    for name, constructor in models.items():
        try:
            model = constructor().fit(disp='off')
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_name = name
        except:
            continue

    return best_name, best_model

def tune_best_vol_model(series, vol_type):
    best_model = None
    best_order = None
    best_aic = np.inf

    for p in range(1, 4):
        for q in range(1, 4):
            try:
                if vol_type == 'GARCH':
                    model = arch_model(series, vol='GARCH', p=p, q=q, mean='Zero').fit(disp='off')
                elif vol_type == 'EGARCH':
                    model = arch_model(series, vol='EGARCH', p=p, q=q, mean='Zero').fit(disp='off')
                elif vol_type == 'GJR-GARCH':
                    model = arch_model(series, vol='GARCH', p=p, o=1, q=q, mean='Zero').fit(disp='off')
                else:
                    continue

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
                    best_model = model
            except:
                continue

    return best_model, best_order

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

    # Fit ARMA(1,1) as example (use your selected order instead)
    try:
        arma_model = ARIMA(series, order=(1, 0, 1)).fit()
    except:
        print("Failed to fit ARMA. Skipping.")
        continue

    residuals = arma_model.resid.dropna()

    # ARCH Effect Test
    arch_pval = test_arch_effects(residuals)
    print(f"ARCH-LM Test p-value: {arch_pval:.4f}")

    if arch_pval > 0.05:
        print("✅ No significant ARCH effect. Skipping volatility modeling.")
        continue

    print("⚠️ ARCH effect detected. Proceeding with GARCH family models.")

    # Compare GARCH families
    best_family, prelim_model = compare_vol_models(residuals)
    print(f"Best volatility family: {best_family}")

    # Tune best family
    final_model, best_order = tune_best_vol_model(residuals, best_family)
    print(f"Best order for {best_family}: {best_order}")
    results.append({
        'Company': company,
        'ARCH_p_value': arch_pval,
        'BestVolModel': best_family,
        'Order': best_order,
        'AIC': final_model.aic
    })

    # Save summary per company
    with open(os.path.join(OUTPUT_FOLDER, f"{company}_{best_family}_summary.txt"), 'w') as f:
        f.write(final_model.summary().as_text())

# Save global summary
summary_df = pd.DataFrame(results)
summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "volatility_model_comparison.csv"), index=False)
print("\n✅ Volatility model analysis complete.")
