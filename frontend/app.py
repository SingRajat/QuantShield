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

# --- CUSTOM CSS: Wall Street Terminal Aesthetic ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* Main App Background & Font */
.stApp {
    background-color: #0A0C0F !important;
    color: #E8E8E8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111316 !important;
    border-right: 1px solid #2A2D35 !important;
    border-top: 1px solid #C9A84C !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #E8E8E8 !important;
    font-weight: 600 !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: #111316 !important;
    border: 1px solid #2A2D35 !important;
    padding: 12px 16px !important;
    border-radius: 2px !important;
}
div[data-testid="stMetricLabel"] {
    color: #8A8A8A !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
div[data-testid="stMetricValue"] {
    color: #E8E8E8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 32px !important;
}

/* Primary Buttons */
button[data-testid="baseButton-primary"] {
    background-color: #C9A84C !important;
    color: #0A0C0F !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
}

/* All Widget Labels */
label, label p, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] {
    color: #D1D5DB !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Input Fields, Selectboxes, Date Pickers */
div[data-baseweb="input"], 
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] * {
    background-color: #16191D !important;
    color: #F3F4F6 !important;
    border-color: #2A2D35 !important;
}

input {
    background-color: #16191D !important;
    color: #F3F4F6 !important;
}

/* File Uploader Container & Dropzone */
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section * {
    background-color: #16191D !important;
    color: #D1D5DB !important;
    border-color: #2A2D35 !important;
}

div[data-testid="stFileUploader"] button {
    background-color: #262B33 !important;
    color: #F3F4F6 !important;
    border: 1px solid #374151 !important;
}

div[data-testid="stFileUploader"] button * {
    color: #F3F4F6 !important;
}

/* Selectbox Dropdown Menu Options */
div[data-baseweb="menu"], ul[role="listbox"], [data-baseweb="popover"] {
    background-color: #16191D !important;
    border: 1px solid #2A2D35 !important;
}
div[data-baseweb="menu"] *, ul[role="listbox"] *, [data-baseweb="popover"] * {
    background-color: #16191D !important;
    color: #F3F4F6 !important;
}
li[role="option"]:hover, [data-baseweb="menu"] div:hover {
    background-color: #262B33 !important;
}

/* Text Area */
textarea {
    background-color: #16191D !important;
    border: 1px solid #2A2D35 !important;
    border-radius: 2px !important;
    color: #F3F4F6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Horizontal line / dividers */
hr {
    border-color: #1A1D21 !important;
}

/* Info Box / AI Insights Box */
div[data-testid="stInfo"] {
    background-color: #1A1D21 !important;
    border: none !important;
    border-left: 2px solid #C9A84C !important;
    color: #E8E8E8 !important;
    border-radius: 0px !important;
}
/* Ensure icon in info box doesn't break styling */
div[data-testid="stInfo"] div.stMarkdown {
    color: #E8E8E8 !important;
}

/* Error/Warning boxes */
div[data-testid="stException"], div[data-testid="stError"] {
    background-color: #1A1D21 !important;
    border-left: 2px solid #B22222 !important;
}
div[data-testid="stWarning"] {
    background-color: #1A1D21 !important;
    border-left: 2px solid #D27D2D !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

import os

# Dynamic API URL for local dev vs production Streamlit Cloud
API_URL = os.getenv("BACKEND_API_URL") or st.secrets.get("BACKEND_API_URL", "http://localhost:8000/api/v1/risk/predict")

st.title("AI-Based Portfolio Risk Monitoring")
st.markdown("<p style='color: #8A8A8A; font-size: 14px;'>Analyze asset-wise exposure and evaluate key risk indicators (volatility, diversification, concentration risk).</p>", unsafe_allow_html=True)

# Sidebar for Portfolio Input
st.sidebar.header("Portfolio Construction")
st.sidebar.markdown("<p style='color: #8A8A8A; font-size: 13px;'>Define the ETF Holdings (tickers must end in '.NS' for Indian equities).</p>", unsafe_allow_html=True)

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

st.sidebar.markdown("<hr style='margin:10px 0; border-color:#2A2D35;'>", unsafe_allow_html=True)
st.sidebar.subheader("Holdings (Tickers & Weights)")
st.sidebar.markdown("<p style='color: #8A8A8A; font-size: 12px; font-family: monospace;'>Format: TICKER.NS, Weight</p>", unsafe_allow_html=True)
holdings_text = st.sidebar.text_area(
    "Holdings Input", 
    value=st.session_state.get("holdings_input", "TCS.NS, 0.5\nINFY.NS, 0.5"),
    height=200,
    label_visibility="collapsed"
)

if st.sidebar.button("Analyze Risk", type="primary", use_container_width=True):
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
                
                # Top Level Result (Badge)
                risk_class = data["risk_class"]
                color_map = {"Low": "#2E8B57", "Medium": "#D27D2D", "High": "#B22222"}
                color = color_map.get(risk_class, "#B22222")
                
                st.markdown(
                    f"""
                    <div style="margin-bottom: 24px;">
                        <span style="background-color: {color}; padding: 4px 10px; border-radius: 2px; color: #FFFFFF; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">
                            {risk_class.upper()} RISK
                        </span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Dashboard explanations
                dashboard_explanations = data.get("dashboard_explanations", {})
                
                # Metrics Cards
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

                st.markdown("---")
                
                # CHART CONFIGURATIONS
                chart_layout_defaults = dict(
                    template="plotly_dark",
                    paper_bgcolor="#111316",
                    plot_bgcolor="#111316",
                    font=dict(family="IBM Plex Sans", color="#E8E8E8", size=12),
                    margin=dict(t=30, b=10, l=10, r=10)
                )

                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("### Portfolio Allocation")
                    df_holdings = pd.DataFrame(holdings_list)
                    fig_alloc = px.pie(
                        df_holdings, 
                        values='weight', 
                        names='ticker', 
                        hole=0.5,
                        color_discrete_sequence=['#C9A84C', '#4A90D9', '#E8E8E8', '#8A8A8A', '#5A5D63']
                    )
                    fig_alloc.update_layout(**chart_layout_defaults, height=320)
                    fig_alloc.update_traces(marker=dict(line=dict(color='#111316', width=2)))
                    st.plotly_chart(fig_alloc, use_container_width=True)
                
                with col_chart2:
                    st.markdown("### Risk Gauge")
                    score = data.get("risk_score", 0)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={'font': {'family': 'IBM Plex Mono', 'color': '#E8E8E8', 'size': 36}},
                        gauge={
                            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#8A8A8A"},
                            'bar': {'color': color, 'thickness': 0.55},
                            'bgcolor': "#1A1D21",
                            'borderwidth': 1,
                            'bordercolor': "#2A2D35",
                            'steps': [
                                {'range': [0, 3.33], 'color': 'rgba(46, 139, 87, 0.2)'},
                                {'range': [3.33, 6.66], 'color': 'rgba(210, 125, 45, 0.2)'},
                                {'range': [6.66, 10.0], 'color': 'rgba(178, 34, 34, 0.2)'}
                            ],
                            'threshold': {
                                'line': {'color': color, 'width': 3},
                                'thickness': 0.75,
                                'value': score
                            }
                        }
                    ))
                    fig_gauge.update_layout(**chart_layout_defaults, height=320)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if "risk_gauge" in dashboard_explanations:
                        st.info(f"💡 **System Insight:** {dashboard_explanations['risk_gauge']}")


                        
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API Backend at {API_URL}. Is FastAPI running?\n\nError: {e}")
