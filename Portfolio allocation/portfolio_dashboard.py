import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Load data
@st.cache_data
def load_returns():
    file_path = Path("forecasted_returns_matrix.csv")
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0)
    else:
        st.error("Forecasted returns matrix not found.")
        return None

returns_df = load_returns()

st.set_page_config(page_title="Tunisian SME Portfolio Dashboard", layout="wide")
st.title(" Tunisian SME Portfolio Allocation Dashboard")

if returns_df is not None:
    companies = returns_df.columns.tolist()
    st.sidebar.title("Investor Profile")
    profile = st.sidebar.radio("Choose your profile:", ["Conservative", "Balanced", "Aggressive"])
    investment = st.sidebar.number_input("Investment Amount (TND):", min_value=1000, value=10000, step=500)

    # Placeholder portfolios (you can replace with real model outputs)
    allocations = {
        "Markowitz": {
            'SIPHA': 0.2476, 'STIP': 0.2421, 'SMD': 0.1378, 'CIL': 0.1139,
            'SPDI': 0.0375, 'STVR': 0.0377, 'WIFAK': 0.0342
        },
        "HRP": {
            'NAKL': 0.7338, 'AMV': 0.0938, 'HANL': 0.0814
        },
        "Risk Parity": {
            'GIF': 0.6485, 'HANL': 0.1223, 'AMV': 0.1356
        },
        "Equal Weight": {company: 1/len(companies) for company in companies}
    }

    # Choose method
    method = st.selectbox("Select Portfolio Strategy:", list(allocations.keys()))
    weights = allocations[method]

    # Compute invested amount per company
    allocation_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    allocation_df['Company'] = allocation_df.index
    allocation_df['Investment (TND)'] = allocation_df['Weight'] * investment

    st.subheader(f"📌 {method} Portfolio Allocation")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(allocation_df[['Company', 'Weight', 'Investment (TND)']].set_index('Company'))

    with col2:
        fig = px.pie(allocation_df, names='Company', values='Investment (TND)',
                     title=f"{method} Portfolio Allocation Pie Chart")
        st.plotly_chart(fig, use_container_width=True)

    # Portfolio metrics placeholder (replace with actual if available)
    st.markdown("---")
    st.subheader("📈 Portfolio Performance Metrics")
    perf_data = {
        'Method': ['Markowitz', 'HRP', 'Risk Parity', 'Equal Weight'],
        'Expected Return': [0.0107, -0.0005, -0.0035, -0.0003],
        'Volatility': [0.0101, 0.0002, 0.0020, 0.0023],
        'Sharpe Ratio': [1.0563, -2.1756, -1.7652, -0.1532]
    }
    perf_df = pd.DataFrame(perf_data).set_index('Method')
    st.dataframe(perf_df)

    perf_fig = px.bar(perf_df, x=perf_df.index, y='Sharpe Ratio',
                      title="Sharpe Ratio Comparison by Strategy",
                      color=perf_df.index, text='Sharpe Ratio')
    st.plotly_chart(perf_fig, use_container_width=True)

    st.markdown("---")
   
