# compare_allocation_strategies.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# STEP 1: MANUAL WEIGHTS (PASTED FROM YOUR RUN)
# ---------------------------
weights = {
    'Markowitz': {
        'BH_xgboost_forecast': 0.0193,
        'CIL_xgboost_forecast': 0.1139,
        'ECYCL_xgboost_forecast': 0.0127,
        'MPBS_xgboost_forecast': 0.0144,
        'NBL_xgboost_forecast': 0.0381,
        'SIPHA_xgboost_forecast': 0.2476,
        'SMD_xgboost_forecast': 0.1378,
        'SOPAT_xgboost_forecast': 0.0442,
        'SPDI_xgboost_forecast': 0.0375,
        'STIP_xgboost_forecast': 0.2421,
        'STVR_xgboost_forecast': 0.0377,
        'WIFAK_xgboost_forecast': 0.0342
    },
    'HRP': {
        'AMV_xgboost_forecast': 0.0938,
        'GIF_xgboost_forecast': 0.0495,
        'HANL_xgboost_forecast': 0.0814,
        'NAKL_xgboost_forecast': 0.7338
    },
    'RiskParity': {
        'AETEC_xgboost_forecast': 0.0303,
        'AMV_xgboost_forecast': 0.1356,
        'CC_xgboost_forecast': 0.0259,
        'GIF_xgboost_forecast': 0.6485,
        'HANL_xgboost_forecast': 0.1223
    },
    'EqualWeight': {}
}

# Fill Equal Weight manually
equal_weight_companies = [
    'AETEC', 'AMV', 'ARTES', 'BHL', 'BH', 'BNA', 'BS', 'BTEI', 'CC', 'CELL', 'CIL',
    'CITY', 'DH', 'ECYCL', 'GIF', 'HANL', 'ICF', 'LNDOR', 'LSTR', 'MIP', 'MNP',
    'MPBS', 'NAKL', 'NBL', 'OTH', 'PLAST', 'SAH', 'SAMAA', 'SERVI', 'SIPHA', 'SITS',
    'SMD', 'SOPAT', 'SOTE', 'SPDI', 'STAR', 'STIP', 'STVR', 'TINV', 'TJL', 'TLNET',
    'TLS', 'TRE', 'UMED', 'WIFAK'
]

equal_weight = 1 / len(equal_weight_companies)
for company in equal_weight_companies:
    weights['EqualWeight'][company + '_xgboost_forecast'] = equal_weight

# ---------------------------
# STEP 2: LOAD FORECASTED RETURNS
# ---------------------------
returns_df = pd.read_csv('forecasted_returns_matrix.csv', index_col=0)

# ---------------------------
# STEP 3: COMPUTE METRICS
# ---------------------------
results = []

for method, w_dict in weights.items():
    # align weights and returns
    w = pd.Series(w_dict)
    r = returns_df[w.index].copy()
    portfolio_returns = (r * w).sum(axis=1)
    
    expected_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = expected_return / volatility if volatility != 0 else 0

    results.append({
        'Method': method,
        'Expected Return': round(expected_return, 4),
        'Volatility': round(volatility, 4),
        'Sharpe Ratio': round(sharpe_ratio, 4)
    })

# ---------------------------
# STEP 4: SHOW RESULTS
# ---------------------------
results_df = pd.DataFrame(results).set_index('Method')
print('\n📊 Portfolio Strategy Comparison:\n')
print(results_df)

# ---------------------------
# STEP 5: PLOTS
# ---------------------------
results_df.plot(kind='bar', figsize=(10, 6), title="Portfolio Strategy Comparison", rot=0)
plt.grid(True)
plt.tight_layout()
plt.savefig("strategy_comparison_metrics.png")
plt.show()
