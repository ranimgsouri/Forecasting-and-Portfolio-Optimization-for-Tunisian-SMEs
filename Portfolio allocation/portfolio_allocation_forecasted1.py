import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import (
    EfficientFrontier, expected_returns, risk_models, HRPOpt,
    EfficientCVaR, BlackLittermanModel, objective_functions
)
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.black_litterman import market_implied_prior_returns
import warnings

warnings.filterwarnings("ignore")

# === Load and clean forecasted returns ===
def load_and_clean_returns(filepath):
    returns = pd.read_csv(filepath, index_col=0)
    
    # Ensure numeric values and handle missing/infinite values
    returns = returns.apply(pd.to_numeric, errors='coerce')
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaNs with column means (better than zeros)
    returns = returns.fillna(returns.mean())
    
    # Winsorize (clip outliers at 5%-95%)
    def winsorize(s):
        l, u = s.quantile([0.05, 0.95])
        return s.clip(lower=l, upper=u)
    
    return returns.apply(winsorize)

# === Robust Optimization Strategies ===

def markowitz(returns, risk_free_rate=0.0):
    mu = expected_returns.mean_historical_return(returns)
    S = CovarianceShrinkage(returns).ledoit_wolf()
    
    # Create new instance for each optimization
    ef = EfficientFrontier(mu, S)
    
    # Add constraints BEFORE optimization
    ef.add_constraint(lambda w: w <= 0.15)  # 15% max allocation
    ef.add_constraint(lambda w: w >= 0)     # No shorting
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    
    try:
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    except Exception as e:
        print(f"Markowitz Sharpe failed: {str(e)}. Falling back to min volatility.")
        try:
            weights = ef.min_volatility()
        except Exception as e:
            print(f"Min volatility also failed: {str(e)}. Using equal weight fallback.")
            return equal_weight(returns)
    
    return ef.clean_weights()

def hrp(returns):
    # Use correlation matrix instead of returns directly
    S = CovarianceShrinkage(returns).ledoit_wolf()
    corr = risk_models.cov_to_corr(S)
    
    # HRP doesn't support constraints directly - filter after optimization
    hrp = HRPOpt(cov_matrix=S)
    weights = hrp.optimize()
    
    # Apply constraints post-optimization
    weights = {k: min(v, 0.15) for k, v in weights.items()}
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

def risk_parity(returns):
    S = CovarianceShrinkage(returns).ledoit_wolf()
    inv_vol = 1 / np.sqrt(np.diag(S))
    weights = inv_vol / inv_vol.sum()
    
    # Apply constraints
    weights = np.clip(weights, 0.01, 0.15)
    weights = weights / weights.sum()
    
    return dict(zip(returns.columns, weights))

def cvar(returns):
    # Need both returns and prices for CVaR
    # Using returns as proxy for prices with fake starting prices
    fake_prices = returns.cumsum() + 100
    try:
        ef_cvar = EfficientCVaR(returns, fake_prices)
        ef_cvar.add_constraint(lambda w: w <= 0.15)
        weights = ef_cvar.min_cvar()
        return ef_cvar.clean_weights()
    except Exception as e:
        print(f"CVaR failed: {str(e)}. Using risk parity fallback.")
        return risk_parity(returns)

def black_litterman(returns):
    S = CovarianceShrinkage(returns).ledoit_wolf()
    
    # Simplified BL model without market caps
    market_prior = expected_returns.mean_historical_return(returns)
    
    # Create simple views (top 3 and bottom 3 assets)
    viewdict = {
        returns.mean().nlargest(3).index[0]: 0.05,
        returns.mean().nsmallest(3).index[0]: -0.05
    }
    
    try:
        bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)
        mu_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        
        ef = EfficientFrontier(mu_bl, S_bl)
        ef.add_constraint(lambda w: w <= 0.15)
        weights = ef.max_sharpe()
        return ef.clean_weights()
    except Exception as e:
        print(f"Black-Litterman failed: {str(e)}. Using Markowitz fallback.")
        return markowitz(returns)

def equal_weight(returns):
    n = returns.shape[1]
    return {col: 1/n for col in returns.columns}

# === Main Execution with Error Handling ===
if __name__ == "__main__":
    returns = load_and_clean_returns("forecasted_returns_matrix.csv")
    
    strategies = [
        ("Markowitz", markowitz),
        ("HRP", hrp),
        ("Risk Parity", risk_parity),
        ("CVaR", cvar),
        ("Black-Litterman", black_litterman),
        ("Equal Weight", equal_weight)
    ]

    results = []
    
    for name, strategy_func in strategies:
        try:
            print(f"\n⚙️ Running {name}...")
            weights = strategy_func(returns)
            
            print(f"\n📌 {name} Allocation (Top Holdings):")
            sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
            for asset, weight in sorted_weights[:10]:  # Show top 10 holdings
                print(f"{asset}: {weight:.2%}")
            
            # Calculate performance metrics
            mu = returns.mean()
            S = CovarianceShrinkage(returns).ledoit_wolf()
            ret = np.dot(mu, pd.Series(weights))
            vol = np.sqrt(np.dot(pd.Series(weights).T, np.dot(S, pd.Series(weights))))
            sharpe = ret / vol if vol > 0 else 0
            
            results.append({
                "Method": name,
                "Expected Return": ret,
                "Volatility": vol,
                "Sharpe Ratio": sharpe
            })
            
            # Plot weights
            weight_series = pd.Series(weights).sort_values(ascending=False)
            
            plt.figure(figsize=(12, 6))
            weight_series[weight_series > 0.01].plot(kind="bar", color='steelblue')
            plt.title(f"{name} Allocation\n(Expected Return: {ret:.2%}, Volatility: {vol:.2%}, Sharpe: {sharpe:.2f})")
            plt.ylabel("Weight")
            plt.xticks(rotation=90)
            plt.ylim(0, min(0.25, weight_series.max() * 1.2))
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"\n❌ Critical failure for {name}: {str(e)}")
            continue
    
    # Print performance comparison table
    print("\n📊 Portfolio Strategy Comparison:\n")
    results_df = pd.DataFrame(results).set_index("Method")
    print(results_df.to_markdown(floatfmt=".4f"))