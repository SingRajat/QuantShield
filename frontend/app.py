import streamlit as st
import requests
import json
import ast
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Risk Monitoring System", layout="wide", page_icon="🛡️")

API_URL = "http://localhost:8000/api/v1/risk/predict"

st.title("AI-Based Portfolio Risk Monitoring")
st.markdown("Analyze asset-wise exposure and evaluate key risk indicators (volatility, diversification, concentration risk).")

# Sidebar for Portfolio Input
st.sidebar.header("Portfolio Construction")
st.sidebar.markdown("Define the ETF Holdings (tickers must end in '.NS' for Indian equities).")

etf_name = st.sidebar.text_input("Portfolio Name", "Custom_Portfolio")
reporting_date = st.sidebar.date_input("Reporting Date").strftime("%Y-%m-%d")

benchmark = st.sidebar.selectbox(
    "Select Benchmark",
    ["^NSEI", "^BSESN", "RELIANCE.NS", "Custom"]
)

if benchmark == "Custom":
    benchmark = st.sidebar.text_input("Enter Custom Ticker")

uploaded_file = st.sidebar.file_uploader(
    "Upload Portfolio CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    csv_holdings_text = "\n".join(
        f"{row['Ticker']},{row['Weight']}"
        for _, row in df.iterrows()
    )
    st.session_state["holdings_input"] = csv_holdings_text

# Simple dataframe or text area for holdings
st.sidebar.subheader("Holdings (Tickers & Weights)")
st.sidebar.markdown("Format: `TICKER.NS, Weight` (e.g., `TCS.NS, 0.5`)")
holdings_text = st.sidebar.text_area(
    "Holdings Input", 
    value=st.session_state.get("holdings_input", "TCS.NS, 0.5\nINFY.NS, 0.5")
)

if st.sidebar.button("Analyze Risk", type="primary"):
    # Parse Holdings
    try:
        holdings_list = []
        lines = holdings_text.strip().split('\n')
        total_weight = 0.0
        for line in lines:
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(f"Invalid format at line: {line}")
            ticker = parts[0].strip()
            weight = float(parts[1].strip())
            holdings_list.append({"ticker": ticker, "weight": weight})
            total_weight += weight
            
        if not (0.95 <= total_weight <= 1.05):
            st.sidebar.warning(f"Weights sum to {total_weight:.2f}. Expected ~1.0")

        # Prepare Payload
        payload = {
            "etf_name": etf_name,
            "reporting_date": reporting_date,
            "holdings": holdings_list,
            "benchmark": benchmark
        }
        
    except Exception as e:
        st.error(f"Error parsing holdings: {e}")
        st.stop()

    with st.spinner("Analyzing portfolio risk..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                
                # Top Level Result
                risk_class = data["risk_class"]
                color = "green" if risk_class == "Low" else "orange" if risk_class == "Medium" else "red"
                st.markdown(f"### 🎯 Predicted Risk Class: **:{color}[{risk_class}]**")
                
                # Dashboard explanations
                dashboard_explanations = data.get("dashboard_explanations", {})
                
                # Metrics Cards
                st.markdown("### Key Risk Indicators")
                metrics = data["metrics"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    with st.container(border=True):
                        vol = metrics.get('Annualized_Volatility', 0)
                        st.metric(label="Annualized Volatility", value=f"{vol:.2%}")
                
                with col2:
                    with st.container(border=True):
                        var = metrics.get('Historical_VaR_95', 0)
                        st.metric(label="Historical VaR (95%)", value=f"{var:.2%}")
                
                with col3:
                    with st.container(border=True):
                        max_dd = metrics.get('Maximum_Drawdown', 0)
                        st.metric(label="Maximum Drawdown", value=f"{max_dd:.2%}")
                
                with col4:
                    with st.container(border=True):
                        div = metrics.get('Diversification_Ratio', 0)
                        st.metric(label="Diversification Ratio", value=f"{div:.2f}")

                st.markdown("---")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("### Portfolio Allocation")
                    df_holdings = pd.DataFrame(holdings_list)
                    fig_alloc = px.pie(df_holdings, values='weight', names='ticker', hole=0.4)
                    fig_alloc.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_alloc, use_container_width=True)
                
                with col_chart2:
                    st.markdown("### Risk Gauge")
                    score = data.get("risk_score", 0)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        gauge={
                            'axis': {'range': [0, 10]},
                            'bar': {'color': color}
                        }
                    ))
                    fig_gauge.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    if "risk_gauge" in dashboard_explanations:
                        st.info(f"💡 **AI Insight:** {dashboard_explanations['risk_gauge']}")

                st.markdown("---")
                st.markdown("### Advanced Analytics")
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    bm_name = data.get("benchmark_name", "^NSEI")
                    st.markdown(f"**Growth: Portfolio vs {bm_name}**")
                    portfolio = data.get("portfolio_returns", [])
                    benchmark_ret = data.get("benchmark_returns", [])
                    
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(y=portfolio, name="Portfolio"))
                    fig_perf.add_trace(go.Scatter(y=benchmark_ret, name=bm_name))
                    fig_perf.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=350)
                    st.plotly_chart(fig_perf, use_container_width=True)
                    if "advanced_analytics" in dashboard_explanations:
                        st.info(f"💡 **AI Insight:** {dashboard_explanations['advanced_analytics']}")
                
                with col_adv2:
                    st.markdown("**Asset Correlation Heatmap**")
                    returns_data = data.get("returns_data", {})
                    if returns_data:
                        returns_df = pd.DataFrame(returns_data)
                        corr = returns_df.corr()
                        fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr, fmt=".2f", cbar=False)
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                    if "asset_correlation" in dashboard_explanations:
                        st.info(f"💡 **AI Insight:** {dashboard_explanations['asset_correlation']}")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API Backend at {API_URL}. Is FastAPI running?\n\nError: {e}")
