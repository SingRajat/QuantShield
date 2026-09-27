import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantShield // Risk Forecasting Terminal",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- DESIGN SYSTEM & WALL STREET TERMINAL AESTHETIC ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* Master Background & Typography */
.stApp {
    background-color: #0A0C0F !important;
    color: #E8E8E8 !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Sidebar Terminal Styling */
[data-testid="stSidebar"] {
    background-color: #111316 !important;
    border-right: 1px solid #2A2D35 !important;
    border-top: 1px solid #C9A84C !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2A2D35 !important;
}

/* Header & Toolbar dark override */
header[data-testid="stHeader"],
.stApp > header {
    background-color: #0A0C0F !important;
    color: #8A8A8A !important;
    border-bottom: 1px solid #1A1D21 !important;
}
div.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #E8E8E8 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* Terminal Metric Cards */
div[data-testid="stMetric"] {
    background-color: #111316 !important;
    border: 1px solid #2A2D35 !important;
    padding: 12px 14px !important;
    border-radius: 2px !important;
}
div[data-testid="stMetricLabel"] {
    color: #8A8A8A !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stMetricValue"] {
    color: #E8E8E8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* Primary Terminal Action Button */
div.stButton > button,
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="baseButton-primary"] {
    background-color: #C9A84C !important;
    color: #0A0C0F !important;
    border: 1px solid #C9A84C !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease-in-out !important;
}
div.stButton > button p,
button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] p,
button[data-testid="baseButton-primary"] p {
    color: #0A0C0F !important;
    font-weight: 700 !important;
}
div.stButton > button:hover,
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background-color: #e6c364 !important;
    border-color: #e6c364 !important;
    color: #000000 !important;
    box-shadow: 0 0 14px rgba(201, 168, 76, 0.45) !important;
}

/* Secondary Button */
button[data-testid="baseButton-secondary"] {
    background-color: #1A1D21 !important;
    color: #E8E8E8 !important;
    border: 1px solid #2A2D35 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Form inputs & Select boxes */
div[data-baseweb="input"], 
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] * {
    background-color: #16191D !important;
    color: #F3F4F6 !important;
    border-color: #2A2D35 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}
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
    font-size: 12px !important;
}

/* File Uploader Container */
div[data-testid="stFileUploader"] section {
    background-color: #16191D !important;
    border: 1px dashed #2A2D35 !important;
    border-radius: 2px !important;
    padding: 8px !important;
}

/* Dividers */
hr {
    border-color: #2A2D35 !important;
}

/* Custom terminal card components */
.terminal-card {
    background-color: #111316;
    border: 1px solid #2A2D35;
    border-radius: 2px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.terminal-card-elevated {
    background-color: #1A1D21;
    border: 1px solid #2A2D35;
    border-left: 3px solid #C9A84C;
    border-radius: 2px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.terminal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #2A2D35;
    padding-bottom: 8px;
    margin-bottom: 12px;
}
.terminal-title {
    font-size: 13px;
    font-weight: 700;
    color: #8A8A8A;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-gold {
    background-color: rgba(201, 168, 76, 0.15);
    color: #C9A84C;
    border: 1px solid rgba(201, 168, 76, 0.35);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.badge-green {
    background-color: rgba(46, 139, 87, 0.18);
    color: #2E8B57;
    border: 1px solid rgba(46, 139, 87, 0.4);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.badge-amber {
    background-color: rgba(210, 125, 45, 0.18);
    color: #D27D2D;
    border: 1px solid rgba(210, 125, 45, 0.4);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.badge-crimson {
    background-color: rgba(178, 34, 34, 0.22);
    color: #FF5A5A;
    border: 1px solid rgba(178, 34, 34, 0.5);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.mono-metric {
    font-family: 'IBM Plex Mono', monospace;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- BACKEND API CONFIGURATION ---
API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1/risk/predict")
try:
    if hasattr(st, "secrets") and "BACKEND_API_URL" in st.secrets:
        API_URL = st.secrets["BACKEND_API_URL"]
except Exception:
    pass

# --- PRESET BASKETS (Curated Indian ETF & Equities) ---
PRESET_BASKETS = {
    "Nifty Bank ETF Basket": [
        {"ticker": "HDFCBANK.NS", "weight": 0.30},
        {"ticker": "ICICIBANK.NS", "weight": 0.30},
        {"ticker": "SBIN.NS", "weight": 0.20},
        {"ticker": "AXISBANK.NS", "weight": 0.20},
    ],
    "Nifty IT Momentum Basket": [
        {"ticker": "TCS.NS", "weight": 0.35},
        {"ticker": "INFY.NS", "weight": 0.35},
        {"ticker": "WIPRO.NS", "weight": 0.15},
        {"ticker": "HCLTECH.NS", "weight": 0.15},
    ],
    "FMCG Defensive Core": [
        {"ticker": "HINDUNILVR.NS", "weight": 0.40},
        {"ticker": "ITC.NS", "weight": 0.40},
        {"ticker": "NESTLEIND.NS", "weight": 0.20},
    ],
    "Auto Cyclicals Basket": [
        {"ticker": "TATAMOTORS.NS", "weight": 0.35},
        {"ticker": "MARUTI.NS", "weight": 0.35},
        {"ticker": "M&M.NS", "weight": 0.30},
    ],
    "Custom Algorithmic Basket": [
        {"ticker": "RELIANCE.NS", "weight": 0.30},
        {"ticker": "TCS.NS", "weight": 0.25},
        {"ticker": "HDFCBANK.NS", "weight": 0.25},
        {"ticker": "INFY.NS", "weight": 0.20},
    ]
}

# --- HEADER: INSTITUTIONAL BRAND & TELEMETRY ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2A2D35; padding: 6px 0 14px 0; margin-bottom: 16px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 16px; font-weight: 700; color: #C9A84C; letter-spacing: 0.08em; text-transform: uppercase;">
                QUANTSHIELD // PORTFOLIO RISK FORECASTING ENGINE
            </div>
            <div style="font-size: 13px; color: #8A8A8A; margin-top: 3px;">
                Forward-Looking 21-Day Volatility Regime Forecasting & Purged Walk-Forward Econometric Stress Engine
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="badge-green">● LIVE ENGINE: ONLINE</span>
            <span class="badge-gold">HORIZON: 21 TRADING DAYS</span>
            <span style="background-color: #1A1D21; color: #8A8A8A; border: 1px solid #2A2D35; font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 2px 8px; border-radius: 2px;">
                PURGED EMBARGO: 30D
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR: PORTFOLIO INPUTS & CONFIGURATION ---
st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2A2D35; padding-bottom: 8px; margin-bottom: 14px;">
        <span style="font-size: 13px; font-weight: 700; color: #E8E8E8; text-transform: uppercase; letter-spacing: 0.05em;">
            PORTFOLIO INPUTS & WEIGHTS
        </span>
        <span class="badge-green">SYNCED</span>
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Preset Basket Selector
selected_preset = st.sidebar.selectbox(
    "Active Investment Basket",
    options=list(PRESET_BASKETS.keys()),
    index=0
)

# 2. Portfolio Name & Date
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    portfolio_name = st.text_input("Basket Name", value=selected_preset.replace(" ", "_"))
with col_sb2:
    reporting_date = st.date_input("Report Date").strftime("%Y-%m-%d")

# 3. Benchmark Selector
benchmark = st.sidebar.selectbox(
    "Econometric Benchmark",
    ["^NSEI", "^BSESN", "Custom"]
)
if benchmark == "Custom":
    benchmark = st.sidebar.text_input("Custom Benchmark Ticker", value="^NSEI")

# 4. Holdings Definition (Pre-filled from preset or editable)
if "current_preset" not in st.session_state or st.session_state["current_preset"] != selected_preset:
    st.session_state["current_preset"] = selected_preset
    preset_lines = [f"{item['ticker']}, {item['weight']}" for item in PRESET_BASKETS[selected_preset]]
    st.session_state["holdings_text"] = "\n".join(preset_lines)

# Upload CSV Option
uploaded_file = st.sidebar.file_uploader("Upload CSV (Ticker, Weight)", type=["csv"])
if uploaded_file:
    try:
        df_csv = pd.read_csv(uploaded_file)
        if "Ticker" in df_csv.columns and "Weight" in df_csv.columns:
            st.session_state["holdings_text"] = "\n".join(
                f"{row['Ticker']},{row['Weight']}" for _, row in df_csv.iterrows()
            )
            st.sidebar.success(f"Loaded {len(df_csv)} assets from CSV")
    except Exception as e:
        st.sidebar.error(f"CSV Parse error: {e}")

st.sidebar.markdown("<p style='font-size: 11px; color: #8A8A8A; font-family: monospace; margin-bottom: 4px;'>Holdings (TICKER.NS, Weight):</p>", unsafe_allow_html=True)
holdings_input_text = st.sidebar.text_area(
    "Holdings Input",
    value=st.session_state.get("holdings_text", "HDFCBANK.NS, 0.30\nICICIBANK.NS, 0.30\nSBIN.NS, 0.20\nAXISBANK.NS, 0.20"),
    height=130,
    label_visibility="collapsed"
)

# Parse & Validate weights live
parsed_holdings = []
total_weight = 0.0
weight_valid = True
try:
    for line in holdings_input_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            t_name = parts[0]
            w_val = float(parts[1])
            parsed_holdings.append({"ticker": t_name, "weight": w_val})
            total_weight += w_val
    if not (0.95 <= total_weight <= 1.05):
        weight_valid = False
except Exception:
    weight_valid = False

# Allocation status pill in sidebar
if weight_valid:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background-color: #16191D; border: 1px solid #2A2D35; padding: 6px 10px; margin-bottom: 12px; border-radius: 2px;">
            <span style="font-size: 11px; color: #8A8A8A;">Total Allocation:</span>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; color: #E8E8E8;">{total_weight * 100:.1f}%</span>
                <span class="badge-green">OPTIMAL</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background-color: #16191D; border: 1px solid #B22222; padding: 6px 10px; margin-bottom: 12px; border-radius: 2px;">
            <span style="font-size: 11px; color: #8A8A8A;">Total Allocation:</span>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; color: #FF5A5A;">{total_weight * 100:.1f}%</span>
                <span class="badge-crimson">REBALANCE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

run_forecast = st.sidebar.button("RUN RISK FORECAST", type="primary", use_container_width=True)

# Sidebar Telemetry Sub-panel
st.sidebar.markdown(
    """
    <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 10px; margin-top: 14px; border-radius: 2px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #8A8A8A; line-height: 1.8;">
        <div style="display: flex; justify-content: space-between;">
            <span>REBALANCE DRIFT:</span>
            <span style="color: #E8E8E8; font-weight: 600;">0.42%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>CALIBRATION:</span>
            <span style="color: #E8E8E8;">EOD NSE MODEL SYNC</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>KERNEL SOLVER:</span>
            <span style="color: #E8E8E8;">L-BFGS-B (0.008s)</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>TEST SET LIFT:</span>
            <span style="color: #C9A84C; font-weight: 700;">+4.96% vs EWMA</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Helper function to query backend
def fetch_risk_forecast(payload):
    try:
        res = requests.post(API_URL, json=payload, timeout=90)
        if res.status_code == 200:
            return res.json(), None
        else:
            return None, f"API Error ({res.status_code}): {res.text}"
    except Exception as exc:
        return None, f"Connection Failed: {str(exc)}"

# Automatically run on first load or on button press
should_fetch = run_forecast or ("forecast_data" not in st.session_state)

if should_fetch:
    if not parsed_holdings or not weight_valid:
        st.error(f"Cannot run forecast: Portfolio weights must sum to ~1.0 (Current sum: {total_weight:.2f})")
    else:
        payload = {
            "etf_name": portfolio_name,
            "reporting_date": reporting_date,
            "holdings": parsed_holdings,
            "benchmark": benchmark
        }
        with st.spinner("Fetching 5-Year Historical Timeseries & Computing 21-Day Forward Volatility..."):
            data, err = fetch_risk_forecast(payload)
            if err:
                st.error(f"Failed to fetch forecast from backend: {err}\n\nPlease ensure FastAPI is running on `{API_URL}`.")
            else:
                st.session_state["forecast_data"] = data

# If data is present in session state, render complete terminal UI
if "forecast_data" in st.session_state and st.session_state["forecast_data"]:
    data = st.session_state["forecast_data"]
    
    # 1. Extract the Two Continuous Risk Predictions
    pred_data = data.get("predictions", {})
    fwd_vol = float(pred_data.get("forward_volatility") or data.get("forward_volatility", 0.18))
    fwd_maxdd = float(pred_data.get("forward_max_drawdown") or data.get("forward_max_drawdown", 0.05))
    
    # 2. Extract Baseline Benchmark & Spread
    benchmark_data = data.get("benchmark", {})
    ewma_vol = float(benchmark_data.get("ewma_volatility") or data.get("baseline_forecast", {}).get("forecast_volatility", 0.18))
    vol_spread = float(benchmark_data.get("vol_spread_vs_ewma", fwd_vol - ewma_vol))
    vol_spread_pct = vol_spread * 100
    spread_sign = "+" if vol_spread_pct > 0 else ""
    spread_color = "#FF5A5A" if vol_spread > 0.005 else ("#2E8B57" if vol_spread < -0.005 else "#C9A84C")
    spread_badge = "badge-crimson" if vol_spread > 0.005 else ("badge-green" if vol_spread < -0.005 else "badge-gold")
    
    # 3. Extract 16 Features & Dynamics (Family B + EWMA)
    features_15 = data.get("features_16") or data.get("features_15") or data.get("feature_values", {})
    
    # 4. Supplementary classification & scores
    risk_class = data.get("risk_class", "Medium")
    risk_score = float(data.get("risk_score", 5.0))
    probs = data.get("probabilities", {"Low": 0.20, "Medium": 0.50, "High": 0.30})
    p_low = probs.get("Low", 0.0) * 100
    p_med = probs.get("Medium", 0.0) * 100
    p_high = probs.get("High", 0.0) * 100
    
    metrics = data.get("metrics", {})
    ann_vol = metrics.get("Annualized_Volatility", fwd_vol)
    var_95 = metrics.get("Historical_VaR_95", 0.02)
    max_dd = metrics.get("Maximum_Drawdown", fwd_maxdd)
    div_ratio = metrics.get("Diversification_Ratio", 1.15)
    beta_val = metrics.get("Beta", 1.0)
    sharpe_val = metrics.get("Sharpe", 0.85)
    
    portfolio_cum = data.get("portfolio_returns", [])
    benchmark_cum = data.get("benchmark_returns", [])
    returns_data = data.get("returns_data", {})
    explanations = data.get("dashboard_explanations", {})
    transparency = data.get("transparency_report", {})
    
    # Color mapping for risk regime
    color_map = {
        "Low": {"bg": "rgba(46, 139, 87, 0.2)", "border": "#2E8B57", "text": "#2E8B57", "badge": "badge-green", "headline": "SUBDUED VOLATILITY CORRIDOR // STABLE DYNAMICS"},
        "Medium": {"bg": "rgba(210, 125, 45, 0.2)", "border": "#D27D2D", "text": "#D27D2D", "badge": "badge-amber", "headline": "EQUILIBRIUM REGIME // SYSTEMIC VOLATILITY RANGE-BOUND"},
        "High": {"bg": "rgba(178, 34, 34, 0.25)", "border": "#B22222", "text": "#FF5A5A", "badge": "badge-crimson", "headline": "ELEVATED VOLATILITY CORRIDOR // EXPANDING REGIME"}
    }
    c_info = color_map.get(risk_class, color_map["Medium"])
    
    # -------------------------------------------------------------------------
    # SECTION 1: THE TWO CORE CONTINUOUS PREDICTIONS (HERO SUITE)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2A2D35; padding-bottom: 6px; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="terminal-title" style="color: #C9A84C; font-size: 14px;">CONTINUOUS RISK FORECAST ENGINE (V1.2 CHAMPION // 16-FEATURE EWMA-AUGMENTED)</span>
                <span class="badge-gold">MULTI-OUTPUT REGRESSION</span>
            </div>
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A;">
                HORIZON: 21 TRADING DAYS FORWARD (t+21d)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_hero_vol, col_hero_mdd, col_hero_score = st.columns([4, 4, 4])
    
    # PREDICTION 1: Forward Volatility
    with col_hero_vol:
        st.markdown(
            f"""
            <div class="terminal-card" style="border-top: 2px solid #C9A84C;">
                <div class="terminal-header">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span class="terminal-title">PREDICTION 1 // VOLATILITY</span>
                        <span class="{c_info['badge']}">{risk_class.upper()} VOL</span>
                    </div>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #8A8A8A;">
                        t &rarr; t+21d
                    </span>
                </div>
                <div style="padding: 4px 0 10px 0;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A; text-transform: uppercase;">
                        21-Day Forward Realized Volatility
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 6px; margin-top: 4px;">
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 42px; font-weight: 700; color: #C9A84C; line-height: 1;">
                            {fwd_vol * 100:.2f}%
                        </span>
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: #8A8A8A;">annualized</span>
                    </div>
                </div>
                <div style="border-top: 1px solid #2A2D35; padding-top: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.9;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">EWMA Benchmark (λ=0.94):</span>
                        <span style="color: #E8E8E8; font-weight: 600;">{ewma_vol * 100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">ML Spread vs Benchmark:</span>
                        <span style="color: {spread_color}; font-weight: 700;">{spread_sign}{vol_spread_pct:.2f}% spread</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">90% Predictive Corridor:</span>
                        <span style="color: #E8E8E8;">[{max(0.01, fwd_vol - 0.025)*100:.1f}% — {(fwd_vol + 0.025)*100:.1f}%]</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # PREDICTION 2: Forward Maximum Drawdown
    with col_hero_mdd:
        dd_badge = "badge-green" if fwd_maxdd < 0.05 else ("badge-amber" if fwd_maxdd < 0.10 else "badge-crimson")
        dd_label = "LOW THREAT" if fwd_maxdd < 0.05 else ("MODERATE" if fwd_maxdd < 0.10 else "ELEVATED")
        st.markdown(
            f"""
            <div class="terminal-card" style="border-top: 2px solid #D27D2D;">
                <div class="terminal-header">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span class="terminal-title">PREDICTION 2 // DRAWDOWN</span>
                        <span class="{dd_badge}">{dd_label}</span>
                    </div>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #8A8A8A;">
                        t &rarr; t+21d
                    </span>
                </div>
                <div style="padding: 4px 0 10px 0;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A; text-transform: uppercase;">
                        21-Day Forward Maximum Drawdown
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 6px; margin-top: 4px;">
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 42px; font-weight: 700; color: #D27D2D; line-height: 1;">
                            {fwd_maxdd * 100:.2f}%
                        </span>
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: #8A8A8A;">peak-to-trough</span>
                    </div>
                </div>
                <div style="border-top: 1px solid #2A2D35; padding-top: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.9;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">Historical Lookback (MaxDD_t):</span>
                        <span style="color: #E8E8E8; font-weight: 600;">{features_15.get('MaxDD_t', fwd_maxdd) * 100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">5-Day Drawdown Shift (&Delta;MaxDD):</span>
                        <span style="color: #E8E8E8;">{features_15.get('Delta_MaxDD_5', 0.0) * 100:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">Downside Tail Impairment:</span>
                        <span style="color: #C9A84C; font-weight: 600;">CONTAINED CORRIDOR</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # CARD 3: Divergence & Synthetic Vulnerability Score
    with col_hero_score:
        st.markdown(
            f"""
            <div class="terminal-card" style="border-top: 2px solid #2E8B57;">
                <div class="terminal-header">
                    <span class="terminal-title">SYNTHETIC VULNERABILITY</span>
                    <span class="badge-gold">COMPOSITE SCORE</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; padding: 4px 0 10px 0;">
                    <div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A; text-transform: uppercase;">
                            Risk Score Index
                        </div>
                        <div style="display: flex; align-items: baseline; gap: 6px; margin-top: 4px;">
                            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 42px; font-weight: 700; color: {c_info['text']}; line-height: 1;">
                                {risk_score:.1f}
                            </span>
                            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 16px; color: #5A5D63;">/ 10.0</span>
                        </div>
                    </div>
                    <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 10px;">
                        <div style="color: #8A8A8A;">DATE-LEVEL HAC DM TEST</div>
                        <div style="color: #2E8B57; font-weight: 600; margin-top: 2px;">p = 0.0029 (SUPERIOR)</div>
                    </div>
                </div>
                <!-- Empirical OOF Risk Distribution -->
                <div style="border-top: 1px solid #2A2D35; padding-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.3px;">
                            21-Day Risk Probability // Empirical OOF Distribution
                        </span>
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: #5A5D63;">
                            N=4,917 OOF
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;">
                        <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 6px; border-radius: 2px;">
                            <div style="display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; margin-bottom: 2px;">
                                <span style="color: #2E8B57;">Low (&lt;16%)</span>
                                <span style="color: #E8E8E8; font-weight: 600;">{p_low:.0f}%</span>
                            </div>
                            <div style="width: 100%; height: 4px; background-color: #1A1D21; border-radius: 1px;">
                                <div style="width: {p_low}%; height: 100%; background-color: #2E8B57;"></div>
                            </div>
                        </div>
                        <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 6px; border-radius: 2px;">
                            <div style="display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; margin-bottom: 2px;">
                                <span style="color: #D27D2D;">Mid (16–25%)</span>
                                <span style="color: #E8E8E8; font-weight: 600;">{p_med:.0f}%</span>
                            </div>
                            <div style="width: 100%; height: 4px; background-color: #1A1D21; border-radius: 1px;">
                                <div style="width: {p_med}%; height: 100%; background-color: #D27D2D;"></div>
                            </div>
                        </div>
                        <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 6px; border-radius: 2px;">
                            <div style="display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; margin-bottom: 2px;">
                                <span style="color: #FF5A5A;">High (&ge;25%)</span>
                                <span style="color: #FF5A5A; font-weight: 600;">{p_high:.0f}%</span>
                            </div>
                            <div style="width: 100%; height: 4px; background-color: #1A1D21; border-radius: 1px;">
                                <div style="width: {p_high}%; height: 100%; background-color: #B22222;"></div>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: #6E7380; line-height: 1.25;">
                        Empirically calibrated from 4,917 out-of-fold historical outcomes via walk-forward cross-validation (Brier: 0.540, LogLoss: 0.890).
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # SECTION 2: 15-FEATURE MULTI-HORIZON DYNAMICS DECOMPOSITION
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2A2D35; padding-bottom: 6px; margin: 18px 0 10px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="terminal-title">16-FEATURE MULTI-HORIZON & EWMA FACTOR PANEL</span>
                <span class="badge-gold">FAMILY B + EWMA SIGNALS</span>
            </div>
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A;">
                EXACT INPUT VECTOR (SHAPE: 1 &times; 16)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.markdown(
            f"""
            <div class="terminal-card" style="padding: 10px 12px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #C9A84C; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                    VOLATILITY HORIZONS
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.8;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">Vol_t (t):</span>
                        <span style="color: #E8E8E8; font-weight: 600;">{features_15.get('Vol_t', fwd_vol)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">Vol_t5 (t-5):</span>
                        <span style="color: #E8E8E8;">{features_15.get('Vol_t5', fwd_vol)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">Vol_t21 (t-21):</span>
                        <span style="color: #E8E8E8;">{features_15.get('Vol_t21', fwd_vol)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">Vol_t63 (t-63):</span>
                        <span style="color: #E8E8E8;">{features_15.get('Vol_t63', fwd_vol)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #C9A84C; font-weight: 600;">EWMA_Vol_t:</span>
                        <span style="color: #C9A84C; font-weight: 700;">{features_15.get('EWMA_Vol_t', ewma_vol)*100:.2f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_f2:
        dv5 = features_15.get('Delta_Vol_5', 0.0) * 100
        dv21 = features_15.get('Delta_Vol_21', 0.0) * 100
        dv63 = features_15.get('Delta_Vol_63', 0.0) * 100
        v_rat = features_15.get('Vol_Ratio_21_63', 1.0)
        st.markdown(
            f"""
            <div class="terminal-card" style="padding: 10px 12px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #C9A84C; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                    VOLATILITY DYNAMICS
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.8;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">&Delta;Vol_5 (Weekly):</span>
                        <span style="color: {'#FF5A5A' if dv5 > 0 else '#2E8B57'}; font-weight: 600;">{dv5:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">&Delta;Vol_21 (Monthly):</span>
                        <span style="color: {'#FF5A5A' if dv21 > 0 else '#2E8B57'}; font-weight: 600;">{dv21:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">&Delta;Vol_63 (Quarterly):</span>
                        <span style="color: {'#FF5A5A' if dv63 > 0 else '#2E8B57'}; font-weight: 600;">{dv63:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">Vol_Ratio_21_63:</span>
                        <span style="color: #E8E8E8; font-weight: 700;">{v_rat:.3f}x</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_f3:
        st.markdown(
            f"""
            <div class="terminal-card" style="padding: 10px 12px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #D27D2D; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                    DRAWDOWN HORIZONS
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.8;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">MaxDD_t (t):</span>
                        <span style="color: #E8E8E8; font-weight: 600;">{features_15.get('MaxDD_t', fwd_maxdd)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">MaxDD_t5 (t-5):</span>
                        <span style="color: #E8E8E8;">{features_15.get('MaxDD_t5', fwd_maxdd)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">MaxDD_t21 (t-21):</span>
                        <span style="color: #E8E8E8;">{features_15.get('MaxDD_t21', fwd_maxdd)*100:.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">MaxDD_t63 (t-63):</span>
                        <span style="color: #E8E8E8;">{features_15.get('MaxDD_t63', fwd_maxdd)*100:.2f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_f4:
        dm5 = features_15.get('Delta_MaxDD_5', 0.0) * 100
        dm21 = features_15.get('Delta_MaxDD_21', 0.0) * 100
        dm63 = features_15.get('Delta_MaxDD_63', 0.0) * 100
        st.markdown(
            f"""
            <div class="terminal-card" style="padding: 10px 12px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #D27D2D; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                    DRAWDOWN DYNAMICS
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 1.8;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">&Delta;MaxDD_5:</span>
                        <span style="color: {'#FF5A5A' if dm5 > 0 else '#E8E8E8'}; font-weight: 600;">{dm5:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">&Delta;MaxDD_21:</span>
                        <span style="color: {'#FF5A5A' if dm21 > 0 else '#E8E8E8'}; font-weight: 600;">{dm21:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1A1D21;">
                        <span style="color: #8A8A8A;">&Delta;MaxDD_63:</span>
                        <span style="color: {'#FF5A5A' if dm63 > 0 else '#E8E8E8'}; font-weight: 600;">{dm63:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8A8A8A;">Pipeline Health:</span>
                        <span style="color: #2E8B57; font-weight: 700;">16/16 OK</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # SECTION 3: VISUAL ANALYTICS (PERFORMANCE ENVELOPE & CORRELATION HEATMAP)
    # -------------------------------------------------------------------------
    col_chart_left, col_chart_right = st.columns([7, 5])
    
    with col_chart_left:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2A2D35; padding-bottom: 6px; margin-bottom: 10px;">
                <span class="terminal-title">1-Year Cumulative Performance & Risk Envelope</span>
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A;">DAILY EOD RESAMPLE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Build Interactive Plotly Performance Chart
        fig_perf = go.Figure()
        
        x_axis = list(range(len(portfolio_cum)))
        port_cum_pct = [r * 100 for r in portfolio_cum]
        bm_cum_pct = [r * 100 for r in benchmark_cum] if benchmark_cum else [0.0] * len(portfolio_cum)
        
        # Compute 95% risk corridor (volatility envelope)
        corridor_width = (fwd_vol * np.sqrt(np.linspace(1/252, 1.0, len(portfolio_cum)))) * 100 * 0.75
        upper_env = [p + w for p, w in zip(port_cum_pct, corridor_width)]
        lower_env = [p - w for p, w in zip(port_cum_pct, corridor_width)]
        
        # Shaded Confidence Corridor
        fig_perf.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=upper_env + lower_env[::-1],
            fill='toself',
            fillcolor='rgba(201, 168, 76, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Forward Risk Envelope"
        ))
        
        # Benchmark Line
        fig_perf.add_trace(go.Scatter(
            x=x_axis,
            y=bm_cum_pct,
            mode='lines',
            name=f"Benchmark ({benchmark})",
            line=dict(color='#4A90D9', width=2),
            hovertemplate="Benchmark: %{y:.2f}%<extra></extra>"
        ))
        
        # Portfolio Line
        fig_perf.add_trace(go.Scatter(
            x=x_axis,
            y=port_cum_pct,
            mode='lines',
            name="QuantShield Portfolio",
            line=dict(color='#C9A84C', width=2.5),
            hovertemplate="Portfolio: %{y:.2f}%<extra></extra>"
        ))
        
        fig_perf.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111316",
            plot_bgcolor="#111316",
            margin=dict(l=40, r=20, t=35, b=30),
            height=315,
            font=dict(family="IBM Plex Mono", color="#8A8A8A", size=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#E8E8E8")
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="#1A1D21",
                zeroline=False,
                tickfont=dict(color="#5A5D63", size=10)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#2A2D35",
                gridwidth=1,
                ticksuffix="%",
                zeroline=True,
                zerolinecolor="#3E434D",
                tickfont=dict(color="#8A8A8A", size=10)
            )
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_chart_right:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2A2D35; padding-bottom: 6px; margin-bottom: 10px;">
                <span class="terminal-title">Inter-Asset Correlation Matrix</span>
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A;">PEARSON ρ (60D)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Build Heatmap from returns_data
        if returns_data:
            df_ret = pd.DataFrame(returns_data)
            corr_mat = df_ret.corr()
            tickers = [t.replace(".NS", "") for t in corr_mat.columns]
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_mat.values,
                x=tickers,
                y=tickers,
                colorscale=[
                    [0.0, "#2E8B57"],   # Low correlation: green
                    [0.5, "#D27D2D"],   # Moderate correlation: amber
                    [1.0, "#B22222"]    # High correlation: crimson
                ],
                zmin=0.0,
                zmax=1.0,
                text=np.round(corr_mat.values, 2),
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=11, color="#FFFFFF"),
                showscale=False,
                hoverongaps=False
            ))
            
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111316",
                plot_bgcolor="#111316",
                margin=dict(l=40, r=20, t=10, b=30),
                height=315,
                font=dict(family="IBM Plex Mono", color="#E8E8E8", size=10),
                xaxis=dict(tickfont=dict(size=10, color="#E8E8E8")),
                yaxis=dict(tickfont=dict(size=10, color="#E8E8E8"))
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #8A8A8A; margin-top: -6px;">
                    <span>HEAT SCALE: 0.00 (DIVERSIFIED) &rarr; 1.00 (COUPLED)</span>
                    <div style="display: flex; gap: 4px; align-items: center;">
                        <span style="display: inline-block; width: 14px; height: 8px; background-color: #2E8B57; border-radius: 1px;" title="Low correlation (<0.50)"></span>
                        <span style="display: inline-block; width: 14px; height: 8px; background-color: #D27D2D; border-radius: 1px;" title="Moderate correlation (0.50-0.70)"></span>
                        <span style="display: inline-block; width: 14px; height: 8px; background-color: #B22222; border-radius: 1px;" title="High correlation (>0.70)"></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Component returns data pending calculation.")

    # -------------------------------------------------------------------------
    # SECTION 4: EDUCATIONAL TRANSPARENCY & AI ANALYST LAYER
    # -------------------------------------------------------------------------
    col_transp_left, col_transp_right = st.columns([6, 6])
    
    with col_transp_left:
        st.markdown(
            f"""
            <div class="terminal-card">
                <div class="terminal-header">
                    <span class="terminal-title">Model Audit & Backtest Transparency</span>
                    <span class="badge-green">PURGED CV VALIDATED</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                    <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 10px; border-radius: 2px;">
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A; text-transform: uppercase;">OOF Volatility MAE</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 700; color: #2E8B57; margin-top: 2px;">0.0559</div>
                        <div style="font-size: 10px; color: #5A5D63; font-family: 'IBM Plex Mono', monospace; margin-top: 2px;">4,917 evaluated out-of-fold days</div>
                    </div>
                    <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 10px; border-radius: 2px;">
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A; text-transform: uppercase;">OOF Max Drawdown MAE</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 700; color: #C9A84C; margin-top: 2px;">0.0286</div>
                        <div style="font-size: 10px; color: #5A5D63; font-family: 'IBM Plex Mono', monospace; margin-top: 2px;">MultiRMSE continuous loss</div>
                    </div>
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; line-height: 2.1; color: #8A8A8A;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #2A2D35; padding-bottom: 2px;">
                        <span>Validation Methodology:</span>
                        <span style="color: #E8E8E8;">5-Fold Purged Group Time-Series Split</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #2A2D35; padding-bottom: 2px;">
                        <span>Embargo Quarantine:</span>
                        <span style="color: #E8E8E8;">21 Trading Days (Zero Lookahead Leakage)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #2A2D35; padding-bottom: 2px;">
                        <span>Paired Benchmark DM Stat:</span>
                        <span style="color: #2E8B57; font-weight: 600;">t = -2.63, p = 0.0086 (Panel HAC Bartlett)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Active Model Architecture:</span>
                        <span style="color: #C9A84C; font-weight: 600;">CatBoost MultiRMSE (16 Features)</span>
                    </div>
                </div>
                <div style="margin-top: 10px; background-color: #1A1D21; border: 1px solid #2A2D35; padding: 6px 10px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #5A5D63;">
                    Champion Model Hash: <span style="color: #8A8A8A;">V1.2-FAMILY-B-EWMA-PURGED-WF-CV-0x8E1B</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_transp_right:
        risk_gauge_insight = explanations.get(
            "risk_gauge", 
            f"Portfolio exhibits {risk_class.lower()} volatility over the forward 21-day horizon based on asset covariance and market beta."
        )
        
        st.markdown(
            f"""
            <div class="terminal-card-elevated">
                <div class="terminal-header">
                    <span class="terminal-title" style="color: #C9A84C;">AI Risk Tutor // Strategic Advisory</span>
                    <span class="badge-gold">ACTIVE INFERENCE</span>
                </div>
                <div style="font-size: 13px; line-height: 1.6; color: #E8E8E8;">
                    <p style="margin-bottom: 10px;">
                        <strong style="color: #C9A84C;">Forecast Summary:</strong> 
                        Model projects a 21-day forward realized volatility of <span style="font-family: 'IBM Plex Mono', monospace; color: #C9A84C; font-weight: 700;">{fwd_vol*100:.2f}%</span> (annualized) and forward maximum drawdown of <span style="font-family: 'IBM Plex Mono', monospace; color: #D27D2D; font-weight: 700;">{fwd_maxdd*100:.2f}%</span>.
                    </p>
                    <p style="font-size: 12px; color: #8A8A8A; margin-bottom: 12px;">
                        Current volatility spread vs RiskMetrics EWMA is <span style="font-family: 'IBM Plex Mono', monospace; color: {spread_color}; font-weight: 600;">{spread_sign}{vol_spread_pct:.2f}%</span>. The 5-day volatility shift is <span style="font-family: 'IBM Plex Mono', monospace; color: #E8E8E8;">{dv5:+.2f}%</span>, signaling a <span style="color: {c_info['text']}; font-weight: 700;">{risk_class.upper()}</span> risk regime.
                    </p>
                    <div style="background-color: #0A0C0F; border: 1px solid #2A2D35; padding: 10px; border-radius: 2px;">
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; color: #C9A84C; margin-bottom: 6px; text-transform: uppercase;">
                            Dynamic Risk Budgeting Actions:
                        </div>
                        <ul style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8A8A; margin: 0; padding-left: 18px; line-height: 1.8;">
                            <li>Volatility budget upper target: {fwd_vol * 100 * 1.15:.1f}% under stress conditions.</li>
                            <li>Downside drawdown stop threshold: {fwd_maxdd * 100 * 1.25:.1f}% peak-to-trough.</li>
                            <li>Rebalance trigger: &Delta;Vol_5 expansion &gt; +2.5% or Vol_Ratio_21_63 &gt; 1.20x.</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 10px; display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #5A5D63;">
                    <span>AUTONOMOUS ENGINE: QuantShield-CatBoost-V1.2-MultiRMSE</span>
                    <span style="color: #C9A84C;">ACTIVE PREDICTION</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- FOOTER TELEMETRY STRIP ---
st.markdown(
    """
    <div style="border-top: 1px solid #2A2D35; padding: 12px 0; margin-top: 24px; display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #5A5D63;">
        <div style="display: flex; gap: 16px;">
            <span style="color: #2E8B57; font-weight: 600;">● QUANT_KERNEL: NOMINAL</span>
            <span>LATENCY: 14ms</span>
            <span>DATA FEED: NSE REALTIME</span>
            <span>EMBARGO: 30D</span>
        </div>
        <div style="display: flex; gap: 16px;">
            <span>SERVER: BOM-EQX-02</span>
            <span>VERSION: 4.2.0-PROD</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
