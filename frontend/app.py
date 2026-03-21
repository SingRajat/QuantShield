import streamlit as st
import requests
import json
import ast

st.set_page_config(page_title="Risk Monitoring System", layout="wide", page_icon="🛡️")

API_URL = "http://localhost:8000/api/v1/risk/predict"

st.title("🛡️ AI-Based Portfolio Risk Monitoring")
st.markdown("Analyze asset-wise exposure and evaluate key risk indicators (volatility, diversification, concentration risk).")

# Sidebar for Portfolio Input
st.sidebar.header("Portfolio Construction")
st.sidebar.markdown("Define the ETF Holdings (tickers must end in '.NS' for Indian equities).")

etf_name = st.sidebar.text_input("Portfolio Name", "Custom_Portfolio")
reporting_date = st.sidebar.date_input("Reporting Date").strftime("%Y-%m-%d")

# Simple dataframe or text area for holdings
st.sidebar.subheader("Holdings (Tickers & Weights)")
st.sidebar.markdown("Format: `TICKER.NS, Weight` (e.g., `TCS.NS, 0.5`)")
holdings_text = st.sidebar.text_area(
    "Holdings Input", 
    "TCS.NS, 0.5\nINFY.NS, 0.5"
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
            "holdings": holdings_list
        }
        
    except Exception as e:
        st.error(f"Error parsing holdings: {e}")
        st.stop()

    with st.spinner("Analyzing portfolio risk via Inference Pipeline..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # Top Level Result
                risk_class = data["risk_class"]
                color = "green" if risk_class == "Low" else "orange" if risk_class == "Medium" else "red"
                st.markdown(f"### 🎯 Predicted Risk Class: **:{color}[{risk_class}]**")
                
                # Explanation
                st.info(f"**🤖 AI Agent Explanation:**\n\n{data['explanation']}")
                
                # Metrics Cards
                st.markdown("### Key Risk Indicators")
                metrics = data["metrics"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    vol = metrics.get('Annualized_Volatility', 0)
                    st.metric(label="Annualized Volatility", value=f"{vol:.2%}")
                
                with col2:
                    var = metrics.get('Historical_VaR_95', 0)
                    st.metric(label="Historical VaR (95%)", value=f"{var:.2%}")
                
                with col3:
                    max_dd = metrics.get('Maximum_Drawdown', 0)
                    st.metric(label="Maximum Drawdown", value=f"{max_dd:.2%}")
                
                with col4:
                    div = metrics.get('Diversification_Ratio', 0)
                    st.metric(label="Diversification Ratio", value=f"{div:.2f}")

            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API Backend at {API_URL}. Is FastAPI running?\n\nError: {e}")
