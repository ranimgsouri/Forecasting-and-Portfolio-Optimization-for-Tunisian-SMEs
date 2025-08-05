import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load forecasted return matrix
df = pd.read_csv("returns_matrix.csv", index_col=0)
df = df.apply(pd.to_numeric, errors="coerce")

# Calculate metrics
mean_returns = df.mean()
volatility = df.std()

# Classify companies based on median thresholds
return_thresh = mean_returns.median()
volatility_thresh = volatility.median()

classification = []
for company in df.columns:
    mu = mean_returns[company]
    sigma = volatility[company]

    if mu >= return_thresh and sigma < volatility_thresh:
        zone = "Gold Zone (High Return, Low Risk)"
    elif mu < return_thresh and sigma < volatility_thresh:
        zone = "Safe Zone (Low Return, Low Risk)"
    elif mu >= return_thresh and sigma >= volatility_thresh:
        zone = "Speculative Zone (High Return, High Risk)"
    else:
        zone = "Danger Zone (Low Return, High Risk)"
    
    classification.append({
        "Company": company.replace("_cleaned", ""),
        "Mean Return": mu,
        "Volatility": sigma,
        "Zone": zone
    })

# Create DataFrame of results
result_df = pd.DataFrame(classification)
result_df.to_csv("advanced_risk_return_matrix1.csv", index=False)

# Set color palette for zones
palette = {
    "Gold Zone (High Return, Low Risk)": "#FFD700",
    "Safe Zone (Low Return, Low Risk)": "#88CCEE",
    "Speculative Zone (High Return, High Risk)": "#FF7F0E",
    "Danger Zone (Low Return, High Risk)": "#D62728"
}

# Plot 1: Risk-Return Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Volatility",
    y="Mean Return",
    hue="Zone",
    palette=palette,
    data=result_df,
    s=100,
    edgecolor='black'
)
for i, row in result_df.iterrows():
    plt.text(row["Volatility"], row["Mean Return"], row["Company"], fontsize=8)

plt.axvline(volatility_thresh, color='gray', linestyle='--')
plt.axhline(return_thresh, color='gray', linestyle='--')
plt.title("Risk vs. Return Classification of Tunisian companies and Private equities")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Forecasted Mean Return")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("risk_return_scatter1.png")
plt.show()

# Plot 2: Zone Distribution Pie Chart
zone_counts = result_df["Zone"].value_counts()
plt.figure(figsize=(7, 7))
zone_counts.plot.pie(autopct="%.1f%%", colors=[palette[z] for z in zone_counts.index])
plt.ylabel("")
plt.title("Distribution of Companies by Risk-Return Classification1")
plt.tight_layout()
plt.savefig("zone_distribution_pie1.png")
plt.show()
