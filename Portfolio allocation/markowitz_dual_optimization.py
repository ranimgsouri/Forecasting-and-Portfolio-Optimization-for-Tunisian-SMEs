import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import (
    EfficientFrontier, 
    expected_returns, 
    risk_models, 
    plotting,
    CovarianceShrinkage,
    HRPOpt,
    objective_functions
)

def load_and_clean_returns(filepath):
    """Ultra-robust returns cleaning"""
    # Load data
    returns = pd.read_csv(
        filepath,
        index_col="date",
        parse_dates=True,
        dayfirst=False
    )
    
    # Convert to numeric and handle extreme values
    returns = returns.apply(pd.to_numeric, errors='coerce')
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Winsorize extreme returns (clip to 5th/95th percentiles)
    def winsorize(s):
        l, u = s.quantile([0.05, 0.95])
        return s.clip(l, u)
    
    returns = returns.apply(winsorize)
    
    # Fill any remaining NaNs with tiny random values
    np.random.seed(42)
    for col in returns.columns:
        mask = returns[col].isna()
        returns.loc[mask, col] = np.random.uniform(-0.0001, 0.0001, size=mask.sum())
    
    # Filter out problematic assets
    valid_cols = returns.columns[
        (returns.abs().max() < 0.5) &  # No single-day returns >50%
        (returns.std() > 0.0001) &     # Minimum volatility
        (returns.std() < 0.1)          # Maximum volatility
    ]
    returns = returns[valid_cols]
    
    return returns

def optimize_portfolio(returns):
    """Multi-strategy optimization"""
    try:
        # Method 1: Classic Markowitz with regularization
        mu = expected_returns.mean_historical_return(returns)
        S = CovarianceShrinkage(returns).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.2))
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Regularization
        
        try:
            weights = ef.max_sharpe(risk_free_rate=0.02)
            print("Optimized using Max Sharpe")
        except:
            weights = ef.min_volatility()
            print("Optimized using Min Volatility")
            
        return ef.clean_weights()
    
    except:
        # Method 2: Hierarchical Risk Parity (more robust)
        print("Falling back to HRP optimization")
        hrp = HRPOpt(returns)
        return hrp.optimize()

# Main execution
if __name__ == "__main__":
    print("Loading and cleaning data...")
    returns = load_and_clean_returns("returns_matrix.csv")
    
    print("\n=== Final Data Quality Check ===")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print(f"Shape: {returns.shape}")
    print(f"Mean returns:\n{returns.mean().describe()}")
    print(f"\nReturn volatilities:\n{returns.std().describe()}")
    
    print("\nOptimizing portfolio...")
    weights = optimize_portfolio(returns)
    
    print("\n📈 Final Portfolio Weights (>1%):")
    for asset, weight in weights.items():
        if weight > 0.01:
            print(f"{asset}: {weight:.2%}")
    
    # Plot results
    plotting.plot_weights(weights)
    plt.title("Portfolio Allocation markowitz")
    plt.show()