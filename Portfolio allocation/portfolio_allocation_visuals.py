import os
import matplotlib.pyplot as plt
import pandas as pd

# === You can use the existing weights dictionary directly, or read from a CSV ===
# Replace this with your actual weights dictionary if needed
allocations = {
    "Markowitz": {
        "BH_xgboost_forecast": 0.0193,
        "CIL_xgboost_forecast": 0.1139,
        "ECYCL_xgboost_forecast": 0.0127,
        "MPBS_xgboost_forecast": 0.0144,
        "NBL_xgboost_forecast": 0.0381,
        "SIPHA_xgboost_forecast": 0.2476,
        "SMD_xgboost_forecast": 0.1378,
        "SOPAT_xgboost_forecast": 0.0442,
        "SPDI_xgboost_forecast": 0.0375,
        "STIP_xgboost_forecast": 0.2421,
        "STVR_xgboost_forecast": 0.0377,
        "WIFAK_xgboost_forecast": 0.0342,
    },
    "HRP": {
        "AMV_xgboost_forecast": 0.0938,
        "GIF_xgboost_forecast": 0.0495,
        "HANL_xgboost_forecast": 0.0814,
        "NAKL_xgboost_forecast": 0.7338,
    },
    "RiskParity": {
        "AETEC_xgboost_forecast": 0.0303,
        "AMV_xgboost_forecast": 0.1356,
        "CC_xgboost_forecast": 0.0259,
        "GIF_xgboost_forecast": 0.6485,
        "HANL_xgboost_forecast": 0.1223,
    },
    "EqualWeight": {
        "AETEC_xgboost_forecast": 0.0222,
        "AMV_xgboost_forecast": 0.0222,
        "ARTES_xgboost_forecast": 0.0222,
        "BHL_xgboost_forecast": 0.0222,
        "BH_xgboost_forecast": 0.0222,
        "BNA_xgboost_forecast": 0.0222,
        "BS_xgboost_forecast": 0.0222,
        "BTEI_xgboost_forecast": 0.0222,
        "CC_xgboost_forecast": 0.0222,
        "CELL_xgboost_forecast": 0.0222,
        "CIL_xgboost_forecast": 0.0222,
        "CITY_xgboost_forecast": 0.0222,
        "DH_xgboost_forecast": 0.0222,
        "ECYCL_xgboost_forecast": 0.0222,
        "GIF_xgboost_forecast": 0.0222,
        "HANL_xgboost_forecast": 0.0222,
        "ICF_xgboost_forecast": 0.0222,
        "LNDOR_xgboost_forecast": 0.0222,
        "LSTR_xgboost_forecast": 0.0222,
        "MIP_xgboost_forecast": 0.0222,
        "MNP_xgboost_forecast": 0.0222,
        "MPBS_xgboost_forecast": 0.0222,
        "NAKL_xgboost_forecast": 0.0222,
        "NBL_xgboost_forecast": 0.0222,
        "OTH_xgboost_forecast": 0.0222,
        "PLAST_xgboost_forecast": 0.0222,
        "SAH_xgboost_forecast": 0.0222,
        "SAMAA_xgboost_forecast": 0.0222,
        "SERVI_xgboost_forecast": 0.0222,
        "SIPHA_xgboost_forecast": 0.0222,
        "SITS_xgboost_forecast": 0.0222,
        "SMD_xgboost_forecast": 0.0222,
        "SOPAT_xgboost_forecast": 0.0222,
        "SOTE_xgboost_forecast": 0.0222,
        "SPDI_xgboost_forecast": 0.0222,
        "STAR_xgboost_forecast": 0.0222,
        "STIP_xgboost_forecast": 0.0222,
        "STVR_xgboost_forecast": 0.0222,
        "TINV_xgboost_forecast": 0.0222,
        "TJL_xgboost_forecast": 0.0222,
        "TLNET_xgboost_forecast": 0.0222,
        "TLS_xgboost_forecast": 0.0222,
        "TRE_xgboost_forecast": 0.0222,
        "UMED_xgboost_forecast": 0.0222,
        "WIFAK_xgboost_forecast": 0.0222,
    }
}

# === Create output folder ===
os.makedirs("plots", exist_ok=True)

# === Plotting function ===
def plot_method(method, weights):
    df = pd.Series(weights).sort_values(ascending=False)

    # Bar chart
    plt.figure(figsize=(10, 6))
    df.plot(kind="bar", color="skyblue")
    plt.title(f"{method} Portfolio Allocation - Bar Chart")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(f"plots/{method}_bar.png")
    plt.close()

    # Pie chart
    plt.figure(figsize=(8, 8))
    df.plot(kind="pie", autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    plt.title(f"{method} Portfolio Allocation - Pie Chart")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"plots/{method}_pie.png")
    plt.close()

# === Loop through methods and plot ===
for method, weights in allocations.items():
    plot_method(method, weights)

print("✅ Portfolio allocation charts saved in 'plots/' folder.")
