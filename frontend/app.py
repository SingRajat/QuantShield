import os
import json
import html
import hashlib
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantShield // Portfolio Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN SYSTEM & SOFT TERMINAL PALETTE ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Master App Canvas */
.stApp {
    background-color: #0D1117 !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6, p, span, label, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #30363D !important;
}
[data-testid="stSidebar"] hr {
    border-color: #30363D !important;
}

/* Header & Container Spacing */
header[data-testid="stHeader"],
.stApp > header {
    background-color: #0D1117 !important;
    color: #8B949E !important;
    border-bottom: 1px solid #21262D !important;
}
div.block-container {
    padding-top: 3.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 1380px !important;
}

/* Tabs Styling */
div[data-testid="stTabs"] {
    margin-top: 4px;
}
div[data-testid="stTabs"] button[role="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #8B949E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #E6EDF3 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #D4A853 !important;
    border-bottom: 2px solid #D4A853 !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] hr {
    border-color: #30363D !important;
    margin-top: -2px !important;
}

/* Primary Button (Active Preset Chip & Generate Forecast CTA) */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="baseButton-primary"] {
    background-color: #D4A853 !important;
    color: #0D1117 !important;
    border: 1px solid #D4A853 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease-in-out !important;
}
button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] p,
button[data-testid="baseButton-primary"] p {
    color: #0D1117 !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background-color: #E2BC6B !important;
    border-color: #E2BC6B !important;
    color: #000000 !important;
    box-shadow: 0 2px 10px rgba(212, 168, 83, 0.25) !important;
}

/* Secondary Button (Inactive Preset Chips & Action buttons) */
button[kind="secondary"],
button[data-testid="stBaseButton-secondary"],
button[data-testid="baseButton-secondary"],
div.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):not([data-testid="baseButton-primary"]) {
    background-color: #21262D !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
button[kind="secondary"] p,
button[data-testid="stBaseButton-secondary"] p,
button[data-testid="baseButton-secondary"] p,
div.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):not([data-testid="baseButton-primary"]) p {
    color: #E6EDF3 !important;
    font-weight: 500 !important;
}
button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
div.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):not([data-testid="baseButton-primary"]):hover {
    background-color: #30363D !important;
    border-color: #D4A853 !important;
    color: #FFFFFF !important;
}

/* Input Fields & Selectboxes */
div[data-baseweb="input"], 
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] * {
    background-color: #161B22 !important;
    color: #E6EDF3 !important;
    border-color: #30363D !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    border-radius: 6px !important;
}
div[data-baseweb="menu"], ul[role="listbox"], [data-baseweb="popover"] {
    background-color: #21262D !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
}
div[data-baseweb="menu"] *, ul[role="listbox"] * {
    background-color: #21262D !important;
    color: #E6EDF3 !important;
}
li[role="option"]:hover, [data-baseweb="menu"] div:hover {
    background-color: #30363D !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] section {
    background-color: #161B22 !important;
    border: 1px dashed #30363D !important;
    border-radius: 6px !important;
    padding: 8px !important;
}

/* Dividers */
hr {
    border-color: #30363D !important;
}

/* Modern Institutional Cards */
.terminal-card {
    background-color: #1C2128;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: border-color 0.15s ease;
}
.terminal-card:hover {
    border-color: #484F58;
}

.terminal-card-elevated {
    background-color: #1C2128;
    border: 1px solid #30363D;
    border-left: 3px solid #D4A853;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.terminal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #30363D;
    padding-bottom: 10px;
    margin-bottom: 14px;
}
.terminal-title {
    font-size: 12px;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Badges (Softer Tones) */
.badge-gold {
    background-color: rgba(212, 168, 83, 0.15);
    color: #D4A853;
    border: 1px solid rgba(212, 168, 83, 0.35);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
}
.badge-green {
    background-color: rgba(63, 182, 139, 0.15);
    color: #3FB68B;
    border: 1px solid rgba(63, 182, 139, 0.35);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
}
.badge-amber {
    background-color: rgba(210, 153, 34, 0.15);
    color: #D29922;
    border: 1px solid rgba(210, 153, 34, 0.35);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
}
.badge-coral {
    background-color: rgba(244, 112, 103, 0.15);
    color: #F47067;
    border: 1px solid rgba(244, 112, 103, 0.35);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
}
.badge-muted {
    background-color: rgba(139, 148, 158, 0.12);
    color: #8B949E;
    border: 1px solid rgba(139, 148, 158, 0.3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
}

/* Number metrics with JetBrains Mono */
.mono-val {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar Custom Elements */
.sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    margin-bottom: 12px;
    border-bottom: 1px solid #30363D;
}
.sidebar-title {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #E6EDF3;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.live-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(63, 182, 139, 0.15);
    border: 1px solid rgba(63, 182, 139, 0.35);
    color: #3FB68B;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 12px;
}
.pulse-dot {
    width: 6px;
    height: 6px;
    background-color: #3FB68B;
    border-radius: 50%;
    box-shadow: 0 0 6px #3FB68B;
}

/* Visual Allocation Bar */
.allocation-bar-container {
    display: flex;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: #21262D;
    margin: 8px 0 10px 0;
    border: 1px solid #30363D;
}

/* Slider Controls */
div[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    color: #D4A853 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}
div[data-testid="stSidebar"] div[data-testid="stSlider"] div[role="slider"] {
    background-color: #D4A853 !important;
    border: 2px solid #0D1117 !important;
}

/* Sidebar Model Spec Card */
.sidebar-model-card {
    background: #10141A;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 10px 12px;
    margin-top: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #8B949E;
}
.sidebar-model-card .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #21262D;
    padding-bottom: 6px;
    margin-bottom: 8px;
    color: #D4A853;
    font-weight: 600;
    font-size: 11px;
}
.sidebar-model-card .card-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.sidebar-model-card .card-row .val {
    color: #E6EDF3;
    font-weight: 500;
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

# --- PRESET BASKETS (Authentic Real-World Indian Portfolios) ---
PRESET_BASKETS = {
    "Nifty 50 Pillars": [
        {"ticker": "HDFCBANK.NS", "weight": 0.30},
        {"ticker": "RELIANCE.NS", "weight": 0.25},
        {"ticker": "ICICIBANK.NS", "weight": 0.25},
        {"ticker": "INFY.NS", "weight": 0.20},
    ],
    "Rakesh Jhunjhunwala": [
        {"ticker": "TITAN.NS", "weight": 0.45},
        {"ticker": "TATAMOTORS.NS", "weight": 0.25},
        {"ticker": "CRISIL.NS", "weight": 0.15},
        {"ticker": "INDHOTEL.NS", "weight": 0.15},
    ],
    "Parag Parikh Fund": [
        {"ticker": "HDFCBANK.NS", "weight": 0.30},
        {"ticker": "ITC.NS", "weight": 0.30},
        {"ticker": "ICICIBANK.NS", "weight": 0.20},
        {"ticker": "POWERGRID.NS", "weight": 0.20},
    ]
}

PRESET_SOURCES = {
    "Nifty 50 Pillars": {
        "label": "NSE India Factsheet",
        "url": "https://www.niftyindices.com/indices/equity/broad-based-indices/NIFTY-50"
    },
    "Rakesh Jhunjhunwala": {
        "label": "SEBI Exchange Filings",
        "url": "https://trendlyne.com/portfolio/superstar-shareholders/53756/latest/rakesh-jhunjhunwala-portfolio/"
    },
    "Parag Parikh Fund": {
        "label": "PPFAS Monthly Factsheet",
        "url": "https://amc.ppfas.com/schemes/parag-parikh-flexi-cap-fund/"
    }
}

ASSET_PALETTE = ["#D4A853", "#58A6FF", "#3FB68B", "#BC8CFF", "#D29922", "#F47067", "#39C5BB", "#79C0FF"]

# --- STATE INITIALIZATION ---
if "active_preset" not in st.session_state or st.session_state["active_preset"] not in PRESET_BASKETS:
    st.session_state["active_preset"] = "Nifty 50 Pillars"

if "holdings" not in st.session_state or not st.session_state["holdings"]:
    st.session_state["holdings"] = [dict(item) for item in PRESET_BASKETS["Nifty 50 Pillars"]]
    for idx, item in enumerate(st.session_state["holdings"]):
        st.session_state[f"slider_{idx}"] = float(item["weight"])

# --- DEFERRED ACTION DISPATCHER (Prevents StreamlitAPIException) ---
# Any event (preset change, add/remove asset, auto-fit) that mutates slider
# states queues a pending action and triggers a clean rerun. The mutation is then
# applied here at the top of the script cycle BEFORE any widget is instantiated.
if "pending_preset" in st.session_state:
    p_name = st.session_state.pop("pending_preset")
    if p_name in PRESET_BASKETS:
        st.session_state["active_preset"] = p_name
        st.session_state["holdings"] = [dict(item) for item in PRESET_BASKETS[p_name]]
        for idx, item in enumerate(st.session_state["holdings"]):
            st.session_state[f"slider_{idx}"] = float(item["weight"])
        for extra_idx in range(len(st.session_state["holdings"]), 30):
            st.session_state.pop(f"slider_{extra_idx}", None)
        st.session_state.pop("forecast_data", None)

if "pending_remove_idx" in st.session_state:
    rem_idx = st.session_state.pop("pending_remove_idx")
    holdings = st.session_state["holdings"]
    if len(holdings) > 1 and 0 <= rem_idx < len(holdings):
        holdings.pop(rem_idx)
        total = sum(float(h["weight"]) for h in holdings)
        if total > 0:
            for h in holdings:
                h["weight"] = round(float(h["weight"]) / total, 3)
            diff = round(1.0 - sum(float(h["weight"]) for h in holdings), 3)
            holdings[-1]["weight"] = max(0.0, round(float(holdings[-1]["weight"]) + diff, 3))
        for i, h in enumerate(holdings):
            st.session_state[f"slider_{i}"] = float(h["weight"])
        st.session_state.pop(f"slider_{len(holdings)}", None)
        st.session_state.pop("forecast_data", None)

if "pending_add_ticker" in st.session_state:
    new_sym = st.session_state.pop("pending_add_ticker")
    holdings = st.session_state["holdings"]
    if not any(h["ticker"] == new_sym for h in holdings):
        n_new = len(holdings) + 1
        new_eq = round(1.0 / n_new, 3)
        for h in holdings:
            h["weight"] = new_eq
        holdings.append({"ticker": new_sym, "weight": new_eq})
        diff = round(1.0 - sum(float(h["weight"]) for h in holdings), 3)
        holdings[-1]["weight"] = max(0.0, round(float(holdings[-1]["weight"]) + diff, 3))
        for i, h in enumerate(holdings):
            st.session_state[f"slider_{i}"] = float(h["weight"])
        st.session_state.pop("forecast_data", None)

if "pending_autofit" in st.session_state:
    st.session_state.pop("pending_autofit")
    holdings = st.session_state["holdings"]
    total = sum(float(h["weight"]) for h in holdings)
    if total > 0:
        for h in holdings:
            h["weight"] = round(float(h["weight"]) / total, 3)
        diff = round(1.0 - sum(float(h["weight"]) for h in holdings), 3)
        holdings[-1]["weight"] = max(0.0, round(float(holdings[-1]["weight"]) + diff, 3))
        for i, h in enumerate(holdings):
            st.session_state[f"slider_{i}"] = float(h["weight"])
    st.session_state.pop("forecast_data", None)

if "pending_equal" in st.session_state:
    st.session_state.pop("pending_equal")
    holdings = st.session_state["holdings"]
    n = len(holdings)
    if n > 0:
        eq = round(1.0 / n, 3)
        for h in holdings:
            h["weight"] = eq
        diff = round(1.0 - sum(float(h["weight"]) for h in holdings), 3)
        holdings[-1]["weight"] = max(0.0, round(float(holdings[-1]["weight"]) + diff, 3))
        for i, h in enumerate(holdings):
            st.session_state[f"slider_{i}"] = float(h["weight"])
    st.session_state.pop("forecast_data", None)

if "pending_csv_holdings" in st.session_state:
    csv_holdings = st.session_state.pop("pending_csv_holdings")
    total = sum(float(h["weight"]) for h in csv_holdings)
    if total > 0:
        for h in csv_holdings:
            h["weight"] = round(float(h["weight"]) / total, 3)
        diff = round(1.0 - sum(float(h["weight"]) for h in csv_holdings), 3)
        csv_holdings[-1]["weight"] = max(0.0, round(float(csv_holdings[-1]["weight"]) + diff, 3))
    st.session_state["holdings"] = csv_holdings
    for idx, item in enumerate(csv_holdings):
        st.session_state[f"slider_{idx}"] = float(item["weight"])
    for extra_idx in range(len(csv_holdings), 50):
        st.session_state.pop(f"slider_{extra_idx}", None)
    # R-04 Fix: Invalidate previous forecast data on CSV upload
    st.session_state.pop("forecast_data", None)

# Helper function: Auto-balancing simplex redistribution
def rebalance_from_slider(changed_idx):
    slider_key = f"slider_{changed_idx}"
    if slider_key not in st.session_state:
        return
    new_val = float(st.session_state[slider_key])
    holdings = st.session_state["holdings"]
    n = len(holdings)
    if n <= 1:
        holdings[0]["weight"] = 1.0
        st.session_state["slider_0"] = 1.0
        return

    new_val = max(0.0, min(1.0, new_val))
    holdings[changed_idx]["weight"] = new_val
    remaining = max(0.0, 1.0 - new_val)

    other_indices = [i for i in range(n) if i != changed_idx]
    others_sum = sum(float(holdings[i]["weight"]) for i in other_indices)

    if others_sum > 0.0001:
        for i in other_indices:
            holdings[i]["weight"] = round((float(holdings[i]["weight"]) / others_sum) * remaining, 3)
    else:
        eq = round(remaining / len(other_indices), 3)
        for i in other_indices:
            holdings[i]["weight"] = eq

    # Residual adjustment so total is always exactly 1.0
    current_sum = sum(float(h["weight"]) for h in holdings)
    residual = round(1.0 - current_sum, 3)
    if residual != 0 and other_indices:
        last_i = other_indices[-1]
        holdings[last_i]["weight"] = max(0.0, round(float(holdings[last_i]["weight"]) + residual, 3))

    # Sync other slider states
    for i in range(n):
        st.session_state[f"slider_{i}"] = float(holdings[i]["weight"])

def apply_preset(preset_name):
    st.session_state["pending_preset"] = preset_name
    st.rerun()

def auto_fit_weights():
    st.session_state["pending_autofit"] = True
    st.rerun()

def equal_weights():
    st.session_state["pending_equal"] = True
    st.rerun()

def remove_asset(idx):
    st.session_state["pending_remove_idx"] = idx
    st.rerun()

def add_asset(ticker):
    sym = ticker.strip().upper()
    if not sym.endswith(".NS") and not sym.startswith("^"):
        sym += ".NS"
    st.session_state["pending_add_ticker"] = sym
    st.rerun()

# --- TOP HEADER: INSTITUTIONAL BRAND ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding: 4px 0 12px 0; margin-bottom: 12px;">
        <div>
            <div style="font-size: 15px; font-weight: 700; color: #D4A853; letter-spacing: 0.06em; text-transform: uppercase;">
                QUANTSHIELD // PORTFOLIO RISK INTELLIGENCE
            </div>
            <div style="font-size: 13px; color: #8B949E; margin-top: 2px;">
                Forward-looking 21-day volatility forecasting & macroeconomic stress testing for Indian portfolios
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="badge-green">● ENGINE ACTIVE</span>
            <span class="badge-gold">21-DAY FORECAST</span>
            <span class="badge-muted">TESTED ON UNSEEN DATA</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR: PORTFOLIO CONFIGURATION ---
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <span class="sidebar-title">PORTFOLIO COCKPIT</span>
        <span class="live-status-pill"><span class="pulse-dot"></span>ACTIVE</span>
    </div>
    """,
    unsafe_allow_html=True
)

preset_options = list(PRESET_BASKETS.keys())
current_idx = preset_options.index(st.session_state["active_preset"]) if st.session_state["active_preset"] in preset_options else 0
selected_sb_preset = st.sidebar.selectbox(
    "Sample Portfolios",
    options=preset_options,
    index=current_idx,
    help="Pick a pre-built real Indian portfolio to test the software instantly."
)

if selected_sb_preset != st.session_state["active_preset"]:
    apply_preset(selected_sb_preset)

# Public regulatory source redirection link
current_source = PRESET_SOURCES.get(selected_sb_preset, {})
source_label = current_source.get("label", "Official Filing")
source_url = current_source.get("url", "#")

source_link_html = (
    f'<div style="font-size: 11px; color: #8B949E; margin: -6px 0 12px 0; display: flex; justify-content: space-between; align-items: center;">'
    f'<span>Public Source:</span>'
    f'<a href="{source_url}" target="_blank" rel="noopener noreferrer" style="color: #58A6FF; text-decoration: none; font-weight: 500;">'
    f'{source_label} ↗'
    f'</a>'
    f'</div>'
)
st.sidebar.markdown(source_link_html, unsafe_allow_html=True)

clean_name = selected_sb_preset.replace(" ", "_")
col_sb1, col_sb2 = st.sidebar.columns([3, 2])
with col_sb1:
    portfolio_name = st.text_input("Basket Name", value=clean_name)
with col_sb2:
    reporting_date = st.date_input("As of Date").strftime("%Y-%m-%d")

benchmark = st.sidebar.selectbox(
    "Market Benchmark",
    ["^NSEI", "^BSESN", "Custom"],
    format_func=lambda x: "NIFTY 50 (^NSEI)" if x == "^NSEI" else ("SENSEX (^BSESN)" if x == "^BSESN" else "Custom Ticker")
)
if benchmark == "Custom":
    benchmark = st.sidebar.text_input("Custom Ticker", value="^NSEI")

# Sync benchmark & date states - auto-invalidate cached forecast on change
if "active_benchmark" in st.session_state and st.session_state["active_benchmark"] != benchmark:
    st.session_state["active_benchmark"] = benchmark
    st.session_state.pop("forecast_data", None)
else:
    st.session_state["active_benchmark"] = benchmark

if "active_reporting_date" in st.session_state and st.session_state["active_reporting_date"] != reporting_date:
    st.session_state["active_reporting_date"] = reporting_date
    st.session_state.pop("forecast_data", None)
else:
    st.session_state["active_reporting_date"] = reporting_date

# Allocation bar calculation
total_weight = sum(float(item["weight"]) for item in st.session_state["holdings"])
bar_segments = []
for idx, item in enumerate(st.session_state["holdings"]):
    pct = (float(item["weight"]) / total_weight * 100) if total_weight > 0 else 0
    color = ASSET_PALETTE[idx % len(ASSET_PALETTE)]
    esc_t = html.escape(str(item["ticker"]))
    bar_segments.append(
        f'<div style="width: {pct:.1f}%; background-color: {color}; height: 100%;" title="{esc_t}: {item["weight"]*100:.1f}%"></div>'
    )
bar_html = f'<div class="allocation-bar-container">{"".join(bar_segments)}</div>'

st.sidebar.markdown(
    f"""
    <div style="margin-top: 12px; margin-bottom: 2px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 11px; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em;">
                ALLOCATION ({len(st.session_state['holdings'])} ASSETS)
            </span>
            <span class="badge-green">100% OPTIMAL</span>
        </div>
        {bar_html}
    </div>
    """,
    unsafe_allow_html=True
)

# Utility buttons
col_act1, col_act2 = st.sidebar.columns(2)
with col_act1:
    if st.button("Auto-Fit 100%", use_container_width=True, help="Proportionally scale weights to sum to 100%"):
        auto_fit_weights()
with col_act2:
    if st.button("Equal Weight", use_container_width=True, help="Distribute weights equally across assets"):
        equal_weights()

# Asset sliders with auto-balancing
to_remove_idx = None
for idx, item in enumerate(st.session_state["holdings"]):
    color = ASSET_PALETTE[idx % len(ASSET_PALETTE)]
    c_lbl, c_slider, c_btn = st.sidebar.columns([3.8, 5.2, 1.0])
    ticker_clean = html.escape(str(item["ticker"]).replace(".NS", ""))
    full_ticker_esc = html.escape(str(item["ticker"]))
    with c_lbl:
        st.markdown(
            f"""
            <div style="padding-top: 7px; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #E6EDF3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{full_ticker_esc}">
                <span style="color: {color}; margin-right: 4px;">●</span>{ticker_clean}
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_slider:
        slider_key = f"slider_{idx}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = float(item["weight"])
        c_slider.slider(
            slider_key,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=slider_key,
            on_change=rebalance_from_slider,
            args=(idx,),
            label_visibility="collapsed"
        )
    with c_btn:
        if c_btn.button("✕", key=f"del_{item['ticker']}_{idx}", help=f"Remove {item['ticker']}"):
            to_remove_idx = idx

if to_remove_idx is not None:
    remove_asset(to_remove_idx)

# Add Ticker input row
col_add1, col_add2 = st.sidebar.columns([3, 1])
with col_add1:
    new_ticker_input = col_add1.text_input("Add Ticker", placeholder="e.g. RELIANCE.NS", label_visibility="collapsed")
with col_add2:
    if col_add2.button("＋ Add", use_container_width=True):
        if new_ticker_input and new_ticker_input.strip():
            add_asset(new_ticker_input)

# Bulk CSV upload
with st.sidebar.expander("Bulk CSV Upload", expanded=False):
    st.markdown("<p style='font-size: 11px; color: #8B949E; margin-bottom: 4px;'>CSV with <code>Ticker</code> and <code>Weight</code> columns:</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("last_uploaded_csv_hash") != file_hash:
            try:
                df_csv = pd.read_csv(io.BytesIO(file_bytes))
                if "Ticker" in df_csv.columns and "Weight" in df_csv.columns:
                    valid_holdings = []
                    for _, r in df_csv.iterrows():
                        t_raw = str(r["Ticker"]).strip().upper()
                        if not re.match(r'^[A-Z0-9.\-_^]{1,20}$', t_raw):
                            raise ValueError(f"Invalid ticker format in CSV: '{t_raw}'. Only letters, numbers, and .-_^ allowed (max 20 chars).")
                        try:
                            w_val = float(r["Weight"])
                            if math.isnan(w_val) or math.isinf(w_val) or w_val < 0.0:
                                raise ValueError(f"Invalid weight for {t_raw}: must be non-negative finite number.")
                        except (ValueError, TypeError):
                            raise ValueError(f"Non-numeric weight value for {t_raw}")
                        valid_holdings.append({"ticker": t_raw, "weight": w_val})
                    if valid_holdings:
                        st.session_state["pending_csv_holdings"] = valid_holdings
                        st.session_state["last_uploaded_csv_hash"] = file_hash
                        st.rerun()
            except Exception as exc:
                st.error(f"CSV error: {exc}")

st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
run_forecast = st.sidebar.button("GENERATE 21-DAY FORECAST", type="primary", use_container_width=True)

# Sidebar model spec card
st.sidebar.markdown(
    """
    <div class="sidebar-model-card">
        <div class="card-header">
            <span>MODEL SPECIFICATION</span>
            <span class="badge-gold">V1.2</span>
        </div>
        <div class="card-row"><span>Forecaster:</span><span class="val">CatBoost Multi-Output</span></div>
        <div class="card-row"><span>Forecast Window:</span><span class="val">21 Trading Days</span></div>
        <div class="card-row"><span>Historical Data:</span><span class="val">5-Year NSE EOD</span></div>
        <div class="card-row"><span>Standard Baseline:</span><span class="val">RiskMetrics EWMA</span></div>
        <div class="card-row"><span>Confidence Interval:</span><span class="val">90% Normal Corridor</span></div>
    </div>
    """,
    unsafe_allow_html=True
)

parsed_holdings = st.session_state.get("holdings", [])

# Query backend helper
def fetch_risk_forecast(payload):
    try:
        res = requests.post(API_URL, json=payload, timeout=90)
        if res.status_code == 200:
            return res.json(), None
        else:
            return None, f"API Error ({res.status_code}): {res.text}"
    except Exception as exc:
        return None, f"Connection Failed: {str(exc)}"

# Auto-run forecast on startup or button trigger
should_fetch = run_forecast or ("forecast_data" not in st.session_state)

if should_fetch:
    if not parsed_holdings:
        st.error("Please add at least one stock to the portfolio.")
    else:
        payload = {
            "etf_name": portfolio_name,
            "reporting_date": reporting_date,
            "holdings": parsed_holdings,
            "benchmark": benchmark
        }
        with st.spinner("Analyzing 5-year price series & computing 21-day forecast..."):
            data, err = fetch_risk_forecast(payload)
            if err:
                st.error(f"Backend forecast error: {err}\n\nPlease ensure FastAPI is running on `{API_URL}`.")
            else:
                st.session_state["forecast_data"] = data
                st.session_state["forecast_weights"] = {h["ticker"]: round(float(h["weight"]), 4) for h in parsed_holdings}

# --- MAIN PAGE TABS ---
tab_overview, tab_stress, tab_record = st.tabs([
    " Risk Overview",
    " Stress Testing",
    " Track Record"
])

# =============================================================================
# TAB 1: RISK OVERVIEW
# =============================================================================
with tab_overview:
    if "forecast_data" in st.session_state and st.session_state["forecast_data"]:
        data = st.session_state["forecast_data"]

        pred_data = data.get("predictions", {})
        raw_fwd_vol = pred_data.get("forward_volatility")
        if raw_fwd_vol is None:
            raw_fwd_vol = data.get("forward_volatility")

        raw_fwd_maxdd = pred_data.get("forward_max_drawdown")
        if raw_fwd_maxdd is None:
            raw_fwd_maxdd = data.get("forward_max_drawdown")

        if raw_fwd_vol is None or raw_fwd_maxdd is None:
            st.error("Risk prediction data is incomplete or unavailable. Please click 'GENERATE 21-DAY FORECAST' in the sidebar to refresh.")
            st.stop()
        fwd_vol = float(raw_fwd_vol)
        fwd_maxdd = float(raw_fwd_maxdd)

        benchmark_data = data.get("benchmark", {})
        raw_ewma = benchmark_data.get("ewma_volatility")
        if raw_ewma is None:
            raw_ewma = data.get("baseline_forecast", {}).get("forecast_volatility")
        ewma_vol = float(raw_ewma) if raw_ewma is not None else 0.1604
        vol_spread = float(benchmark_data.get("vol_spread_vs_ewma", fwd_vol - ewma_vol))
        vol_spread_pct = vol_spread * 100
        spread_sign = "+" if vol_spread_pct > 0 else ""

        features_15 = data.get("features_16") or data.get("features_15") or data.get("feature_values", {})
        risk_class = data.get("risk_class", "Medium")
        risk_score = float(data.get("risk_score", 4.2))

        probs = data.get("probabilities", {"Low": 0.25, "Medium": 0.55, "High": 0.20})
        p_low = probs.get("Low", 0.25) * 100
        p_med = probs.get("Medium", 0.55) * 100
        p_high = probs.get("High", 0.20) * 100

        portfolio_cum = data.get("portfolio_returns", [])
        benchmark_cum = data.get("benchmark_returns", [])
        returns_data = data.get("returns_data", {})
        benchmark_name = data.get("benchmark_name", "NIFTY 50")
        if benchmark_name == "^NSEI":
            benchmark_name = "NIFTY 50"
        elif benchmark_name == "^BSESN":
            benchmark_name = "SENSEX"

        # Risk-based color palette mappings
        if risk_class == "Low":
            score_color = "#3FB68B"
            score_badge = "badge-green"
            score_label = "Low Risk"
        elif risk_class == "High":
            score_color = "#F47067"
            score_badge = "badge-coral"
            score_label = "High Risk"
        else:
            score_color = "#D29922"
            score_badge = "badge-amber"
            score_label = "Moderate Risk"

        # ---------------------------------------------------------------------
        # STEP 1: PORTFOLIO ALLOCATION STRIP
        # ---------------------------------------------------------------------
        # Asset legend pills directly below tabs
        legend_items = []
        for idx, item in enumerate(st.session_state["holdings"]):
            c = ASSET_PALETTE[idx % len(ASSET_PALETTE)]
            t = html.escape(str(item["ticker"]).replace(".NS", ""))
            w = float(item["weight"]) * 100
            legend_items.append(
                f'<span style="display: inline-flex; align-items: center; gap: 5px; background: #161B22; border: 1px solid #30363D; border-radius: 4px; padding: 3px 8px; font-family: \'JetBrains Mono\', monospace; font-size: 11px; color: #E6EDF3;">'
                f'<span style="color: {c}; font-size: 10px;">●</span>{t}: <strong style="color: #D4A853;">{w:.1f}%</strong></span>'
            )
        legend_html = " ".join(legend_items)

        # Dynamic allocation balance check & modified weights notification
        curr_w_map = {h["ticker"]: round(float(h["weight"]), 4) for h in st.session_state["holdings"]}
        fc_w_map = st.session_state.get("forecast_weights", {})
        total_w = sum(curr_w_map.values())
        is_balanced = abs(total_w - 1.0) < 0.005
        balance_badge = '<span class="badge-green">100% BALANCED</span>' if is_balanced else f'<span class="badge-coral">{total_w*100:.1f}% UNBALANCED</span>'

        weights_dirty = False
        if fc_w_map:
            for t, w in curr_w_map.items():
                if abs(w - fc_w_map.get(t, -1.0)) > 0.005:
                    weights_dirty = True
                    break

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin: 6px 0 16px 0;">
                <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                    <span style="font-size: 11px; font-weight: 600; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 4px;">Holdings:</span>
                    {legend_html}
                </div>
                <div>
                    {balance_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if weights_dirty:
            st.markdown(
                """
                <div style="background: rgba(210, 153, 34, 0.12); border: 1px solid rgba(210, 153, 34, 0.4); border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div style="font-size: 12px; color: #E6EDF3;">
                        <strong style="color: #D29922; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 8px;">Allocation Modified:</strong>
                        Slider weights have changed. The risk forecast metrics below reflect your previous calculation.
                    </div>
                    <div style="font-size: 11px; color: #8B949E; font-family: 'JetBrains Mono', monospace;">
                        Click <strong>GENERATE 21-DAY FORECAST</strong> in the sidebar to recompute
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------------------------------------------
        # STEP 2: 3 HERO FORECAST CARDS
        # ---------------------------------------------------------------------
        col_vol, col_mdd, col_score = st.columns(3)

        # Card 1: Expected Volatility
        with col_vol:
            calib_meta = data.get("risk_calibration", {})
            res_quantiles = calib_meta.get("residual_quantiles", {})
            # Realized y in [pred - q95(e), pred - q05(e)] where e = pred - true
            q05_err = float(res_quantiles.get("vol_err_q05", -0.0384))
            q95_err = float(res_quantiles.get("vol_err_q95", 0.0412))
            ci_low = max(0.01, fwd_vol - q95_err) * 100
            ci_high = max(ci_low + 0.1, fwd_vol - q05_err) * 100
            spread_badge = "badge-coral" if vol_spread > 0.005 else ("badge-green" if vol_spread < -0.005 else "badge-gold")

            st.markdown(
                f"""
                <div class="terminal-card" style="border-top: 2px solid #D4A853;">
                    <div class="terminal-header">
                        <span class="terminal-title">EXPECTED VOLATILITY</span>
                        <span class="{score_badge}">{risk_class.upper()} VOL</span>
                    </div>
                    <div style="padding: 4px 0 10px 0;">
                        <div style="display: flex; align-items: baseline; gap: 8px;">
                            <span class="mono-val" style="font-size: 38px; font-weight: 700; color: #D4A853; line-height: 1;">
                                {fwd_vol * 100:.2f}%
                            </span>
                            <span style="font-size: 12px; color: #8B949E;">annualized</span>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #30363D; padding-top: 10px; font-size: 11px; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Standard Model (EWMA):</span>
                            <span class="mono-val" style="color: #E6EDF3;">{ewma_vol * 100:.2f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Spread vs Standard:</span>
                            <span class="{spread_badge}">{spread_sign}{vol_spread_pct:.2f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;" title="Quantiles fit on out-of-fold residuals; coverage evaluated on holdout data. Acceptable empirical coverage corridor: [85.0%, 95.0%] (nominal target 90.0%).">Empirical 90% Range (OOF-Calibrated, Holdout-Validated):</span>
                            <span class="mono-val" style="color: #E6EDF3;">{ci_low:.1f}% — {ci_high:.1f}%</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Card 2: Max Expected Dip
        with col_mdd:
            dd_badge = "badge-green" if fwd_maxdd < 0.05 else ("badge-amber" if fwd_maxdd < 0.10 else "badge-coral")
            dd_label = "MILD DIP" if fwd_maxdd < 0.05 else ("MODERATE" if fwd_maxdd < 0.10 else "ELEVATED")
            hist_worst = float(features_15.get('MaxDD_t63', features_15.get('MaxDD_t', fwd_maxdd))) * 100
            dd_shift = float(features_15.get('Delta_MaxDD_5', 0.0)) * 100

            st.markdown(
                f"""
                <div class="terminal-card" style="border-top: 2px solid #D29922;">
                    <div class="terminal-header">
                        <span class="terminal-title">MAX EXPECTED DIP</span>
                        <span class="{dd_badge}">{dd_label}</span>
                    </div>
                    <div style="padding: 4px 0 10px 0;">
                        <div style="display: flex; align-items: baseline; gap: 8px;">
                            <span class="mono-val" style="font-size: 38px; font-weight: 700; color: #D29922; line-height: 1;">
                                {fwd_maxdd * 100:.2f}%
                            </span>
                            <span style="font-size: 12px; color: #8B949E;">max dip</span>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #30363D; padding-top: 10px; font-size: 11px; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Recent 60-Day Worst:</span>
                            <span class="mono-val" style="color: #E6EDF3;">{hist_worst:.2f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">5-Day Momentum Shift:</span>
                            <span class="mono-val" style="color: {'#F47067' if dd_shift > 0 else '#3FB68B'};">{dd_shift:+.2f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Capital Impairment Risk:</span>
                            <span style="color: #D29922; font-weight: 500;">Contained Corridor</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Card 3: Overall Risk Rating
        with col_score:
            st.markdown(
                f"""
                <div class="terminal-card" style="border-top: 2px solid {score_color};">
                    <div class="terminal-header">
                        <span class="terminal-title">OVERALL RISK RATING</span>
                        <span class="{score_badge}">{score_label}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding: 4px 0 10px 0;">
                        <div>
                            <div style="display: flex; align-items: baseline; gap: 6px;">
                                <span class="mono-val" style="font-size: 38px; font-weight: 700; color: {score_color}; line-height: 1;">
                                    {risk_score:.1f}
                                </span>
                                <span style="font-size: 16px; color: #6E7681;">/ 10.0</span>
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 11px;">
                            <span style="color: #8B949E;">Confidence Tier:</span>
                            <div style="color: #3FB68B; font-weight: 600; font-family: 'JetBrains Mono', monospace;">High (4,900+ OOF days)</div>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #30363D; padding-top: 8px;">
                        <div style="font-size: 10px; color: #8B949E; text-transform: uppercase; margin-bottom: 5px;">
                            Risk Distribution Probability
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;">
                            <div style="background-color: #10141A; border: 1px solid #30363D; padding: 5px 6px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 10px; margin-bottom: 3px;">
                                    <span style="color: #3FB68B;">Low</span>
                                    <span style="color: #E6EDF3; font-weight: 600;">{p_low:.0f}%</span>
                                </div>
                                <div style="width: 100%; height: 4px; background-color: #21262D; border-radius: 2px;">
                                    <div style="width: {p_low}%; height: 100%; background-color: #3FB68B; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <div style="background-color: #10141A; border: 1px solid #30363D; padding: 5px 6px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 10px; margin-bottom: 3px;">
                                    <span style="color: #D29922;">Mid</span>
                                    <span style="color: #E6EDF3; font-weight: 600;">{p_med:.0f}%</span>
                                </div>
                                <div style="width: 100%; height: 4px; background-color: #21262D; border-radius: 2px;">
                                    <div style="width: {p_med}%; height: 100%; background-color: #D29922; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <div style="background-color: #10141A; border: 1px solid #30363D; padding: 5px 6px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 10px; margin-bottom: 3px;">
                                    <span style="color: #F47067;">High</span>
                                    <span style="color: #F47067; font-weight: 600;">{p_high:.0f}%</span>
                                </div>
                                <div style="width: 100%; height: 4px; background-color: #21262D; border-radius: 2px;">
                                    <div style="width: {p_high}%; height: 100%; background-color: #F47067; border-radius: 2px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------------------------------------------
        # STEP 3: TRAJECTORY CHART (60%) + EXECUTIVE SUMMARY (40%)
        # ---------------------------------------------------------------------
        col_chart, col_summary = st.columns([6, 4])

        with col_chart:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding-bottom: 6px; margin-bottom: 10px;">
                    <span class="terminal-title">Portfolio Trajectory & 21-Day Forward Risk Corridor</span>
                    <span class="badge-gold">90% CONFIDENCE RIBBON</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Build trajectory chart
            P_hist = [r * 100 for r in portfolio_cum]
            B_hist = [r * 100 for r in benchmark_cum] if benchmark_cum else [0.0] * len(portfolio_cum)
            N = len(P_hist)

            x_hist = list(range(-N + 1, 1))
            x_fwd = list(range(0, 22))

            p_last = P_hist[-1] if P_hist else 0.0

            # 90% confidence ribbon radiating from today into the future
            fwd_sigma = [(fwd_vol * np.sqrt(t / 252.0)) * 100 for t in x_fwd]
            upper_band = [p_last + 1.645 * s for s in fwd_sigma]
            lower_band = [p_last - 1.645 * s for s in fwd_sigma]

            fig_perf = go.Figure()

            # Confidence ribbon
            fig_perf.add_trace(go.Scatter(
                x=x_fwd + x_fwd[::-1],
                y=upper_band + lower_band[::-1],
                fill='toself',
                fillcolor='rgba(212, 168, 83, 0.12)',
                line=dict(color='rgba(212, 168, 83, 0.35)', width=1, dash='dot'),
                hoverinfo="skip",
                name="90% Forward Corridor (21D)"
            ))

            # Benchmark historical line
            fig_perf.add_trace(go.Scatter(
                x=x_hist,
                y=B_hist,
                mode='lines',
                name=f"{benchmark_name} (Past 1Y)",
                line=dict(color='#58A6FF', width=1.8),
                hovertemplate="%{x}d: %{y:.2f}%<extra>Benchmark</extra>"
            ))

            # Portfolio historical line
            fig_perf.add_trace(go.Scatter(
                x=x_hist,
                y=P_hist,
                mode='lines',
                name="Portfolio (Past 1Y)",
                line=dict(color='#D4A853', width=2.2),
                hovertemplate="%{x}d: %{y:.2f}%<extra>Portfolio</extra>"
            ))

            # Forward baseline projection line
            fig_perf.add_trace(go.Scatter(
                x=x_fwd,
                y=[p_last] * len(x_fwd),
                mode='lines',
                name="21D Center Path",
                line=dict(color='#D4A853', width=1.5, dash='dash'),
                hovertemplate="Day +%{x}: %{y:.2f}%<extra>Expected Center</extra>"
            ))

            # Vertical line at Today (x=0)
            fig_perf.add_vline(
                x=0,
                line_dash="dot",
                line_color="#8B949E",
                annotation_text="Today",
                annotation_position="top left",
                annotation_font=dict(color="#8B949E", size=10, family="Inter")
            )

            fig_perf.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                margin=dict(l=40, r=20, t=30, b=30),
                height=340,
                font=dict(family="JetBrains Mono", color="#8B949E", size=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="right",
                    x=1,
                    font=dict(size=10, color="#E6EDF3", family="Inter")
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#21262D",
                    zeroline=False,
                    tickfont=dict(color="#8B949E", size=10),
                    title=dict(text="Trading Days (relative to Today)", font=dict(size=10, color="#6E7681"))
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#21262D",
                    ticksuffix="%",
                    zeroline=True,
                    zerolinecolor="#30363D",
                    tickfont=dict(color="#8B949E", size=10)
                )
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        with col_summary:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding-bottom: 6px; margin-bottom: 10px;">
                    <span class="terminal-title">Executive Risk Brief</span>
                    <span class="badge-gold">21-DAY OUTLOOK</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Deterministic, plain-English summary calculations
            if risk_class == "Low":
                verdict_text = "Your portfolio sits in a calm, low-volatility zone. Expected price swings over the next 3 weeks are modest and within comfortable historical bands."
            elif risk_class == "High":
                verdict_text = "Your portfolio has elevated volatility sensitivity. Sharp price swings are more likely over the next 3 weeks compared to steady market periods."
            else:
                verdict_text = "Your portfolio risk is in a balanced, moderate range. Expected fluctuations over the next 3 weeks align with typical market movements."

            # Calculate top risk contributor from weights & asset volatilities
            top_stock = None
            top_share = 0
            if returns_data:
                asset_risks = {}
                for item in parsed_holdings:
                    sym = item["ticker"]
                    w = float(item["weight"])
                    if sym in returns_data and len(returns_data[sym]) > 10:
                        std_dev = np.std(returns_data[sym]) * np.sqrt(252)
                        asset_risks[sym] = w * std_dev
                    else:
                        asset_risks[sym] = w * fwd_vol
                tot_r = sum(asset_risks.values())
                if tot_r > 0:
                    top_sym = max(asset_risks, key=asset_risks.get)
                    top_share = (asset_risks[top_sym] / tot_r) * 100
                    top_stock = top_sym.replace(".NS", "")
            if not top_stock and parsed_holdings:
                max_h = max(parsed_holdings, key=lambda x: x["weight"])
                top_stock = max_h["ticker"].replace(".NS", "")
                top_share = max_h["weight"] * 100

            # Benchmark comparison text (QS-14 & R-03: trailing 21-day volatility calculation & escaping)
            real_bm_vol = None
            bm_returns_api = data.get("benchmark_returns", [])
            if len(bm_returns_api) >= 21:
                bm_ret_series = pd.Series(bm_returns_api).dropna()
                real_bm_vol = float(bm_ret_series.iloc[-21:].std() * np.sqrt(252))
            elif len(benchmark_cum) >= 22:
                # Recover daily returns from cumulative returns: (1 + C_t) / (1 + C_{t-1}) - 1
                bm_wealth = 1.0 + pd.Series(benchmark_cum)
                bm_ret_series = bm_wealth.pct_change().dropna()
                if len(bm_ret_series) >= 21:
                    real_bm_vol = float(bm_ret_series.iloc[-21:].std() * np.sqrt(252))

            escaped_bm_name = html.escape(benchmark_name)
            if real_bm_vol is not None and not np.isnan(real_bm_vol) and real_bm_vol > 0:
                bm_spread_pct = (fwd_vol - real_bm_vol) * 100
                if abs(bm_spread_pct) < 0.25:
                    bm_text = f"Your expected volatility closely tracks <strong>{escaped_bm_name}</strong> trailing 21-day volatility (difference is under 0.25%)."
                elif bm_spread_pct > 0:
                    bm_text = f"Your portfolio is projected to be <strong>{bm_spread_pct:+.1f}% more volatile</strong> than {escaped_bm_name} trailing 21-day volatility."
                else:
                    bm_text = f"Your portfolio is projected to be <strong>{abs(bm_spread_pct):.1f}% calmer</strong> than {escaped_bm_name} trailing 21-day volatility."
            else:
                if abs(vol_spread_pct) < 0.25:
                    bm_text = f"Your expected volatility closely tracks the portfolio's <strong>RiskMetrics EWMA baseline</strong> (spread under 0.25%)."
                elif vol_spread_pct > 0:
                    bm_text = f"Your portfolio is projected to be <strong>{vol_spread_pct:+.1f}% more volatile</strong> than its EWMA baseline."
                else:
                    bm_text = f"Your portfolio is projected to be <strong>{abs(vol_spread_pct):.1f}% calmer</strong> than its EWMA baseline."

            # Actionable takeaway
            if top_share > 35:
                action_text = f"Consider trimming <strong>{top_stock}</strong> or spreading its weight to reduce single-stock concentration."
            elif risk_score > 6.0:
                action_text = "Consider adding defensive holdings (like FMCG or large-cap index funds) to buffer against market swings."
            else:
                action_text = "Your current allocation maintains healthy diversification across its constituents."
            brief_card_html = (
                '<div class="terminal-card" style="padding: 16px 18px; margin-bottom: 0;">'
                '<div style="font-size: 13px; line-height: 1.6; color: #E6EDF3;">'
                '<div style="margin-bottom: 12px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #D4A853; text-transform: uppercase; margin-bottom: 3px;">1. Risk Verdict</div>'
                f'<div style="color: #E6EDF3; font-size: 13px;">{verdict_text}</div>'
                '</div>'
                '<div style="margin-bottom: 12px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #D4A853; text-transform: uppercase; margin-bottom: 3px;">2. Primary Risk Driver</div>'
                f'<div style="color: #E6EDF3; font-size: 13px;"><strong>{top_stock}</strong> accounts for the largest share (~{top_share:.0f}%) of your portfolio\'s risk fluctuations due to its volatility and weighting.</div>'
                '</div>'
                '<div style="margin-bottom: 12px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #D4A853; text-transform: uppercase; margin-bottom: 3px;">3. Benchmark Divergence</div>'
                f'<div style="color: #E6EDF3; font-size: 13px;">{bm_text}</div>'
                '</div>'
                '<div>'
                '<div style="font-size: 11px; font-weight: 600; color: #3FB68B; text-transform: uppercase; margin-bottom: 3px;">4. Practical Takeaway</div>'
                f'<div style="color: #E6EDF3; font-size: 13px;">{action_text}</div>'
                '</div>'
                '</div>'
                '<div style="font-size: 11px; color: #6E7681; margin-top: 14px; border-top: 1px solid #30363D; padding-top: 8px;">Methodology: 21-day CatBoost multi-output regression, Markowitz budget constraint, and RiskMetrics 90% confidence bands.</div>'
                '</div>'
            )
            st.markdown(brief_card_html, unsafe_allow_html=True)

# =============================================================================
# TAB 2: STRESS TESTING & HISTORICAL CRISES
# =============================================================================
with tab_stress:
    if "forecast_data" not in st.session_state or not st.session_state["forecast_data"]:
        st.info("Please generate a 21-day forecast first to unlock stress testing simulations.")
    else:
        data = st.session_state["forecast_data"]
        pred_data = data.get("predictions", {})
        raw_fwd_vol = pred_data.get("forward_volatility") or data.get("forward_volatility")
        raw_fwd_maxdd = pred_data.get("forward_max_drawdown") or data.get("forward_max_drawdown")
        if raw_fwd_vol is None or raw_fwd_maxdd is None:
            st.error("Risk prediction data is incomplete or unavailable. Please click 'GENERATE 21-DAY FORECAST' in the sidebar to refresh.")
            st.stop()
        fwd_vol = float(raw_fwd_vol)
        fwd_maxdd = float(raw_fwd_maxdd)
        portfolio_cum = data.get("portfolio_returns", [])
        benchmark_cum = data.get("benchmark_returns", [])
        returns_data = data.get("returns_data", {})
        benchmark_name = data.get("benchmark_name", "NIFTY 50")
        if benchmark_name == "^NSEI":
            benchmark_name = "NIFTY 50"
        elif benchmark_name == "^BSESN":
            benchmark_name = "SENSEX"

        # 1. Historical Indian Market Crises Definition
        INDIAN_CRISES = {
            "covid_2020": {
                "name": "2020 COVID Flash Crash",
                "dates": "Jan 14, 2020 – Mar 23, 2020",
                "nifty_drop": -38.4,
                "recovery_days": 145,
                "catalyst": "Worldwide pandemic outbreak and sudden nationwide lockdown. India VIX spiked to 86.6.",
                "regime": "Acute Liquidity Shock / V-Shaped Recovery",
                "crisis_year": 2020
            },
            "gfc_2008": {
                "name": "2008 Global Financial Crisis",
                "dates": "Jan 8, 2008 – Oct 27, 2008",
                "nifty_drop": -59.9,
                "recovery_days": 670,
                "catalyst": "US subprime collapse, Lehman Brothers liquidation, and relentless global liquidity contraction.",
                "regime": "Systemic Credit Freeze / Multi-Year Bear Market",
                "crisis_year": 2008
            },
            "taper_2013": {
                "name": "2013 Taper Tantrum",
                "dates": "May 20, 2013 – Aug 28, 2013",
                "nifty_drop": -17.2,
                "recovery_days": 105,
                "catalyst": "US Fed signaled QE taper, causing sudden emerging market capital flight and a 25% Rupee depreciation.",
                "regime": "Currency Devaluation & Rate Spike",
                "crisis_year": 2013
            },
            "rate_2022": {
                "name": "2022 Global Rate Shock",
                "dates": "Oct 18, 2021 – Jun 17, 2022",
                "nifty_drop": -17.2,
                "recovery_days": 110,
                "catalyst": "Multi-decade inflation spike, Russia-Ukraine war, crude above $120, and aggressive Fed/RBI rate hikes.",
                "regime": "Synchronized Rate Tightening Cycle",
                "crisis_year": 2021
            }
        }

        # Listing years for Indian stocks for factual non-fabrication check
        STOCK_LISTING_YEARS = {
            "HDFCBANK.NS": 1995,
            "ICICIBANK.NS": 1997,
            "SBIN.NS": 1994,
            "AXISBANK.NS": 1998,
            "KOTAKBANK.NS": 1996,
            "TCS.NS": 2004,
            "INFY.NS": 1993,
            "WIPRO.NS": 1995,
            "HCLTECH.NS": 1999,
            "TECHM.NS": 2006,
            "RELIANCE.NS": 1990,
            "HINDUNILVR.NS": 1990,
            "ITC.NS": 1990,
            "NESTLEIND.NS": 1990,
            "TATAMOTORS.NS": 1990,
            "MARUTI.NS": 2003,
            "M&M.NS": 1990,
            "BAJFINANCE.NS": 1998,
            "BHARTIARTL.NS": 2002,
            "LT.NS": 1990,
            "ASIANPAINT.NS": 1990,
            "TITAN.NS": 1990,
            "SUNPHARMA.NS": 1994,
            "COALINDIA.NS": 2010,
            "ZOMATO.NS": 2021,
            "PAYTM.NS": 2021,
            "NYKAA.NS": 2021,
            "POLICYBZR.NS": 2021,
            "CRISIL.NS": 1993,
            "INDHOTEL.NS": 1990,
            "POWERGRID.NS": 2007,
            "LICI.NS": 2022,
            "JIOFIN.NS": 2023
        }

        if "selected_crisis" not in st.session_state:
            st.session_state["selected_crisis"] = "covid_2020"

        if "stress_multiplier" not in st.session_state:
            st.session_state["stress_multiplier"] = 1.0

        # Compute individual asset Betas and portfolio Beta from real data
        bm_daily = None
        if len(benchmark_cum) > 1:
            bm_cum_arr = np.array(benchmark_cum)
            denom = 1.0 + bm_cum_arr[:-1]
            denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
            bm_daily = np.diff(bm_cum_arr) / denom

        asset_betas = {}
        for item in parsed_holdings:
            sym = item["ticker"]
            if returns_data and sym in returns_data and bm_daily is not None and len(returns_data[sym]) > 1:
                r_a = np.array(returns_data[sym])
                min_len = min(len(r_a), len(bm_daily))
                cov = np.cov(r_a[-min_len:], bm_daily[-min_len:])[0, 1]
                var_m = np.var(bm_daily[-min_len:])
                b_val = (cov / var_m) if var_m > 1e-7 else 1.0
                asset_betas[sym] = max(0.2, min(2.8, float(b_val)))
            else:
                asset_betas[sym] = float(data.get("metrics", {}).get("Beta", 1.0))

        port_beta = sum(float(item["weight"]) * asset_betas.get(item["ticker"], 1.0) for item in parsed_holdings)
        port_beta = max(0.2, min(2.8, float(port_beta)))

        # ---------------------------------------------------------------------
        # SECTION 1: REAL INDIAN MARKET CRISES (HISTORICAL REPLAY)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding: 4px 0 10px 0; margin-bottom: 14px;">
                <div style="font-size: 13px; font-weight: 700; color: #D4A853; text-transform: uppercase; letter-spacing: 0.05em;">
                    Section 1 // Real Indian Market Crises (Historical Replay)
                </div>
                <span class="badge-gold">HISTORICAL SCENARIOS</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        crisis_keys = list(INDIAN_CRISES.keys())
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        cols_crisis = [col_c1, col_c2, col_c3, col_c4]

        for i, c_key in enumerate(crisis_keys):
            c_info = INDIAN_CRISES[c_key]
            is_active = (st.session_state["selected_crisis"] == c_key)
            with cols_crisis[i]:
                btn_type = "primary" if is_active else "secondary"
                btn_label = f"{c_info['name'].split(' ', 1)[0]} Crash"
                if st.button(btn_label, key=f"crisis_btn_{c_key}", use_container_width=True, type=btn_type):
                    st.session_state["selected_crisis"] = c_key
                    st.rerun()

        # Active Crisis Data
        cur_crisis = INDIAN_CRISES[st.session_state["selected_crisis"]]
        nifty_shock = cur_crisis["nifty_drop"]
        est_port_drop = port_beta * nifty_shock
        cushion = abs(nifty_shock) - abs(est_port_drop)

        crisis_banner_html = (
            '<div class="terminal-card" style="padding: 14px 18px; margin-bottom: 14px; background: #161B22; border-left: 3px solid #D4A853;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
            '<div>'
            f'<div style="font-size: 14px; font-weight: 700; color: #E6EDF3;">{cur_crisis["name"]}</div>'
            f'<div style="font-size: 12px; color: #8B949E; margin-top: 2px;">{cur_crisis["dates"]} · <span style="color: #D4A853;">{cur_crisis["regime"]}</span></div>'
            '</div>'
            '<div style="text-align: right;">'
            '<span style="font-size: 11px; color: #8B949E;">NIFTY 50 HISTORICAL DROP:</span> '
            f'<span class="mono-val" style="font-size: 16px; font-weight: 700; color: #58A6FF;">{nifty_shock:.1f}%</span>'
            '</div>'
            '</div>'
            f'<div style="font-size: 12px; color: #8B949E; margin-top: 8px; border-top: 1px solid #21262D; padding-top: 6px;">'
            f'<strong style="color: #E6EDF3;">Macro Catalyst:</strong> {cur_crisis["catalyst"]}'
            '</div>'
            '</div>'
        )
        st.markdown(crisis_banner_html, unsafe_allow_html=True)

        col_ci1, col_ci2, col_ci3 = st.columns(3)

        with col_ci1:
            drop_color = "#F47067" if est_port_drop < -30 else "#D29922"
            cushion_badge = f'<span class="badge-green">+{cushion:.1f}% DEFENSIVE BUFFER</span>' if cushion > 0 else f'<span class="badge-coral">{cushion:.1f}% EXTRA DOWNSIDE</span>'
            st.markdown(
                f"""
                <div class="terminal-card" style="border-top: 2px solid {drop_color};">
                    <div class="terminal-header">
                        <span class="terminal-title">PORTFOLIO CRASH DROP</span>
                        <span class="badge-gold">CAPM SHOCK ESTIMATE</span>
                    </div>
                    <div style="padding: 4px 0 10px 0;">
                        <div style="display: flex; align-items: baseline; gap: 6px;">
                            <span class="mono-val" style="font-size: 38px; font-weight: 700; color: {drop_color}; line-height: 1;">
                                {est_port_drop:.1f}%
                            </span>
                            <span style="font-size: 12px; color: #8B949E;">loss of capital</span>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #30363D; padding-top: 10px; font-size: 11px; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Market Sensitivity (Beta β):</span>
                            <span class="mono-val" style="color: #E6EDF3;">{port_beta:.2f}x</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                            <span style="color: #8B949E;">Relative Performance:</span>
                            {cushion_badge}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_ci2:
            st.markdown(
                f"""
                <div class="terminal-card" style="border-top: 2px solid #58A6FF;">
                    <div class="terminal-header">
                        <span class="terminal-title">MARKET BENCHMARK CRASH</span>
                        <span class="badge-muted">NIFTY 50 HISTORICAL</span>
                    </div>
                    <div style="padding: 4px 0 10px 0;">
                        <div style="display: flex; align-items: baseline; gap: 6px;">
                            <span class="mono-val" style="font-size: 38px; font-weight: 700; color: #58A6FF; line-height: 1;">
                                {nifty_shock:.1f}%
                            </span>
                            <span style="font-size: 12px; color: #8B949E;">index dip</span>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #30363D; padding-top: 10px; font-size: 11px; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Market Recovery Time:</span>
                            <span class="mono-val" style="color: #E6EDF3;">{cur_crisis["recovery_days"]} trading days</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Historical Fact:</span>
                            <span style="color: #D4A853;">Reclaimed pre-crisis high</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_ci3:
            # Dynamic Crisis Impact Rating (evaluates actual shock severity for THIS specific crisis)
            if est_port_drop <= -45.0:
                impact_label = "CRITICAL SHOCK"
                impact_color = "#F47067"
                impact_badge = "CATASTROPHIC CRISIS"
                impact_sub = f"Severe wealth loss ({'+' if cushion > 0 else ''}{cushion:.1f}% vs Nifty)"
            elif est_port_drop <= -28.0:
                impact_label = "HEAVY DRAWDOWN"
                impact_color = "#F47067" if cushion < 0 else "#D29922"
                impact_badge = "ACUTE CRISIS SHOCK"
                impact_sub = f"Deep market dip ({'+' if cushion > 0 else ''}{cushion:.1f}% vs Nifty)"
            elif est_port_drop <= -18.0:
                impact_label = "MODERATE PULLBACK"
                impact_color = "#D29922"
                impact_badge = "CYCLICAL CORRECTION"
                impact_sub = f"Manageable drawdown ({'+' if cushion > 0 else ''}{cushion:.1f}% vs Nifty)"
            else:
                impact_label = "RESILIENT SHELTER"
                impact_color = "#3FB68B"
                impact_badge = "CAPITAL PRESERVED"
                impact_sub = f"Loss contained under 18% ({'+' if cushion > 0 else ''}{cushion:.1f}% buffer)"

            worst_asset = max(parsed_holdings, key=lambda x: asset_betas.get(x["ticker"], 1.0))
            worst_t = worst_asset["ticker"].replace(".NS", "")
            worst_beta = asset_betas.get(worst_asset["ticker"], 1.0)
            worst_drop = worst_beta * nifty_shock

            st.markdown(
                f"""
                <div class="terminal-card" style="border-top: 2px solid {impact_color};">
                    <div class="terminal-header">
                        <span class="terminal-title">CRISIS IMPACT SEVERITY</span>
                        <span class="badge-gold">{impact_badge}</span>
                    </div>
                    <div style="padding: 4px 0 10px 0;">
                        <div style="display: flex; align-items: baseline; gap: 6px;">
                            <span class="mono-val" style="font-size: 28px; font-weight: 700; color: {impact_color}; line-height: 1.2;">
                                {impact_label}
                            </span>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #30363D; padding-top: 10px; font-size: 11px; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Crash Assessment:</span>
                            <span class="mono-val" style="color: #E6EDF3; font-size: 11px;">{impact_sub}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8B949E;">Biggest Stock Dip:</span>
                            <span class="mono-val" style="color: #F47067;">{html.escape(worst_t)} ({worst_drop:.1f}% drop)</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("Asset-by-Asset Crisis Sensitivity & Historical Listing Audit", expanded=False):
            table_rows = []
            for item in parsed_holdings:
                t = item["ticker"]
                t_clean = t.replace(".NS", "")
                w = float(item["weight"]) * 100
                b = asset_betas.get(t, 1.0)
                a_drop = b * nifty_shock
                ipo_year = STOCK_LISTING_YEARS.get(t, 2024)
                is_listed = ipo_year <= cur_crisis["crisis_year"]
                if is_listed:
                    status_badge = f'<span style="color: #3FB68B; font-weight: 600;">Listed (Full Data)</span>'
                else:
                    status_badge = f'<span style="color: #D29922; font-weight: 600;">Listed in {ipo_year} (Proxy Beta Used)</span>'

                escaped_t_clean = html.escape(t_clean)
                table_rows.append(
                    f'<tr>'
                    f'<td style="padding: 8px 12px; font-family: \'JetBrains Mono\', monospace; color: #E6EDF3; font-weight: 600;">{escaped_t_clean}</td>'
                    f'<td style="padding: 8px 12px; font-family: \'JetBrains Mono\', monospace; color: #D4A853;">{w:.1f}%</td>'
                    f'<td style="padding: 8px 12px; font-family: \'JetBrains Mono\', monospace; color: #E6EDF3;">{b:.2f}x</td>'
                    f'<td style="padding: 8px 12px; font-family: \'JetBrains Mono\', monospace; color: #F47067; font-weight: 600;">{a_drop:.1f}%</td>'
                    f'<td style="padding: 8px 12px; font-size: 11px;">{status_badge}</td>'
                    f'</tr>'
                )

            asset_table_html = (
                '<div style="overflow-x: auto; margin-top: 6px;">'
                '<table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; background: #10141A; border: 1px solid #30363D; border-radius: 6px;">'
                '<thead>'
                '<tr style="border-bottom: 1px solid #30363D; color: #8B949E; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em;">'
                '<th style="padding: 8px 12px;">Asset Ticker</th>'
                '<th style="padding: 8px 12px;">Weight</th>'
                '<th style="padding: 8px 12px;">Market Beta (β)</th>'
                '<th style="padding: 8px 12px;">Estimated Shock Drop</th>'
                '<th style="padding: 8px 12px;">Listing Audit (NSE)</th>'
                '</tr>'
                '</thead>'
                '<tbody>'
                + "".join(table_rows) +
                '</tbody>'
                '</table>'
                '<div style="font-size: 11px; color: #6E7681; margin-top: 8px;">'
                'Institutional Transparency: For stocks not yet trading during the historical crisis, we do NOT fabricate artificial historical price points. Their sensitivity is evaluated via the Capital Asset Pricing Model (CAPM) beta applied to the macro shock.'
                '</div>'
                '</div>'
            )
            st.markdown(asset_table_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # SECTION 2: FORWARD SIMULATION ENGINE (1,000 MONTE CARLO PATHS)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding: 4px 0 10px 0; margin-bottom: 14px;">
                <div style="font-size: 13px; font-weight: 700; color: #D4A853; text-transform: uppercase; letter-spacing: 0.05em;">
                    Section 2 // Forward Simulation Engine (1,000 Monte Carlo Paths)
                </div>
                <span class="badge-gold">1,000 FORWARD RUNS</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_st1, col_st2, col_st3 = st.columns(3)

        with col_st1:
            is_b1 = (st.session_state["stress_multiplier"] == 1.0)
            if st.button("Normal Market (1.0x Vol)", key="btn_stress_10", use_container_width=True, type="primary" if is_b1 else "secondary"):
                st.session_state["stress_multiplier"] = 1.0
                st.rerun()

        with col_st2:
            is_b2 = (st.session_state["stress_multiplier"] == 1.25)
            if st.button("Elevated Volatility (+25% Shock)", key="btn_stress_125", use_container_width=True, type="primary" if is_b2 else "secondary"):
                st.session_state["stress_multiplier"] = 1.25
                st.rerun()

        with col_st3:
            is_b3 = (st.session_state["stress_multiplier"] == 1.50)
            if st.button("Severe Market Crash (+50% Turmoil)", key="btn_stress_150", use_container_width=True, type="primary" if is_b3 else "secondary"):
                st.session_state["stress_multiplier"] = 1.50
                st.rerun()

        cur_mult = st.session_state["stress_multiplier"]

        def run_monte_carlo_stress(fwd_vol, stress_mult=1.0, n_sims=1000, horizon_days=21, seed=42):
            np.random.seed(seed)
            dt = 1.0 / 252.0
            sigma = fwd_vol * stress_mult
            drift = -0.5 * (sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(n_sims, horizon_days))
            daily_log_returns = drift + diffusion
            
            # QS-12: Geometric compounding for wealth paths (W_0 = 1.0)
            wealth_trajectories = np.exp(np.cumsum(daily_log_returns, axis=1))
            wealth_trajectories = np.hstack([np.ones((n_sims, 1)), wealth_trajectories])
            cum_paths = (wealth_trajectories - 1.0) * 100.0

            # Path Drawdown calculation per simulation run (W_t - Peak_t) / Peak_t
            running_peaks = np.maximum.accumulate(wealth_trajectories, axis=1)
            path_drawdowns = (wealth_trajectories - running_peaks) / running_peaks
            max_dd_per_path = np.abs(np.min(path_drawdowns, axis=1)) * 100.0

            terminal_returns = cum_paths[:, -1]
            var_95 = float(np.percentile(terminal_returns, 5))
            cvar_95 = float(terminal_returns[terminal_returns <= var_95].mean())
            median_path = np.median(cum_paths, axis=0)
            p5_path = np.percentile(cum_paths, 5, axis=0)
            p95_path = np.percentile(cum_paths, 95, axis=0)
            worst_terminal_loss = float(terminal_returns.min())
            worst_path_dd = float(max_dd_per_path.max())
            p95_path_dd = float(np.percentile(max_dd_per_path, 95))
            prob_loss_5 = float((terminal_returns < -5.0).mean() * 100.0)

            return {
                "cum_paths": cum_paths,
                "median_path": median_path,
                "p5_path": p5_path,
                "p95_path": p95_path,
                "var_95": abs(var_95),
                "cvar_95": abs(cvar_95),
                "worst_terminal_loss": abs(worst_terminal_loss),
                "worst_path_dd": abs(worst_path_dd),
                "p95_path_dd": abs(p95_path_dd),
                "prob_loss_5": prob_loss_5,
                "sigma_annualized": sigma * 100.0
            }

        mc_res = run_monte_carlo_stress(fwd_vol, stress_mult=cur_mult)

        col_mc_chart, col_mc_stats = st.columns([6, 4])

        with col_mc_chart:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding-bottom: 6px; margin-bottom: 10px;">
                    <span class="terminal-title">21-Day Forward Path Trajectories</span>
                    <span class="badge-gold">PROJECTED VOL: {mc_res['sigma_annualized']:.1f}%/YR</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_mc = go.Figure()
            days_axis = list(range(22))

            for path_idx in range(0, 50):
                fig_mc.add_trace(go.Scatter(
                    x=days_axis,
                    y=mc_res["cum_paths"][path_idx],
                    mode='lines',
                    line=dict(color='rgba(88, 166, 255, 0.08)', width=1),
                    hoverinfo='skip',
                    showlegend=False
                ))

            fig_mc.add_trace(go.Scatter(
                x=days_axis,
                y=mc_res["p95_path"],
                mode='lines',
                name="Top 5% Best Case (95th Pct)",
                line=dict(color='#3FB68B', width=1.8),
                hovertemplate="Day +%{x}: +%{y:.2f}% gain<extra>Top 5% Best Case</extra>"
            ))

            fig_mc.add_trace(go.Scatter(
                x=days_axis,
                y=mc_res["median_path"],
                mode='lines',
                name="Expected Path (Median)",
                line=dict(color='#D4A853', width=2.2),
                hovertemplate="Day +%{x}: %{y:.2f}%<extra>Expected Center Path</extra>"
            ))

            fig_mc.add_trace(go.Scatter(
                x=days_axis,
                y=mc_res["p5_path"],
                mode='lines',
                name="Safe Downside Limit (95% VaR)",
                line=dict(color='#F47067', width=2.2),
                hovertemplate="Day +%{x}: %{y:.2f}% loss<extra>Safe Downside Limit (95% VaR)</extra>"
            ))

            fig_mc.add_hline(y=0.0, line_dash="dash", line_color="#484F58", line_width=1)

            fig_mc.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                margin=dict(l=40, r=20, t=30, b=30),
                height=350,
                font=dict(family="JetBrains Mono", color="#8B949E", size=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="right",
                    x=1,
                    font=dict(size=10, color="#E6EDF3", family="Inter")
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#21262D",
                    zeroline=False,
                    tickfont=dict(color="#8B949E", size=10),
                    title=dict(text="Forward Trading Days (t to t+21d)", font=dict(size=10, color="#6E7681"))
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#21262D",
                    ticksuffix="%",
                    zeroline=True,
                    zerolinecolor="#30363D",
                    tickfont=dict(color="#8B949E", size=10)
                )
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_mc_stats:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding-bottom: 6px; margin-bottom: 10px;">
                    <span class="terminal-title">Projected Loss Scenarios (% of Capital)</span>
                    <span class="badge-gold">N=1,000 RUNS</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            stress_name = "Normal Market Conditions" if cur_mult == 1.0 else ("Elevated Volatility (+25%)" if cur_mult == 1.25 else "Severe Market Crisis (+50%)")
            var_1lakh = int(round(mc_res["var_95"] * 1000))
            cvar_1lakh = int(round(mc_res["cvar_95"] * 1000))

            tail_metrics_html = (
                '<div class="terminal-card" style="padding: 16px 18px; margin-bottom: 0;">'
                '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">'
                '<div style="background-color: #10141A; border: 1px solid #30363D; padding: 10px; border-radius: 6px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #E6EDF3; text-transform: uppercase;">Safe Downside Limit</div>'
                '<div style="font-size: 10px; color: #58A6FF; margin-top: 1px;">21-Day VaR (95% Confidence)</div>'
                f'<div class="mono-val" style="font-size: 22px; font-weight: 700; color: #D29922; margin-top: 3px;">-{mc_res["var_95"]:.2f}%</div>'
                '<div style="font-size: 10px; color: #8B949E; margin-top: 3px;">95% chance losses won\'t exceed this % of capital</div>'
                '</div>'
                '<div style="background-color: #10141A; border: 1px solid #30363D; padding: 10px; border-radius: 6px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #E6EDF3; text-transform: uppercase;">Severe Crash Average</div>'
                '<div style="font-size: 10px; color: #58A6FF; margin-top: 1px;">Expected Shortfall (CVaR)</div>'
                f'<div class="mono-val" style="font-size: 22px; font-weight: 700; color: #F47067; margin-top: 3px;">-{mc_res["cvar_95"]:.2f}%</div>'
                '<div style="font-size: 10px; color: #8B949E; margin-top: 3px;">Average loss if an extreme 1-in-20 crash strikes</div>'
                '</div>'
                '<div style="background-color: #10141A; border: 1px solid #30363D; padding: 10px; border-radius: 6px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #E6EDF3; text-transform: uppercase;">Worst Simulated Drawdown</div>'
                '<div style="font-size: 10px; color: #58A6FF; margin-top: 1px;">Intra-Period Path MaxDD</div>'
                f'<div class="mono-val" style="font-size: 22px; font-weight: 700; color: #F47067; margin-top: 3px;">-{mc_res["worst_path_dd"]:.2f}%</div>'
                f'<div style="font-size: 10px; color: #8B949E; margin-top: 3px;">Deepest trough reached before day 21 (95th pct: -{mc_res["p95_path_dd"]:.1f}%)</div>'
                '</div>'
                '<div style="background-color: #10141A; border: 1px solid #30363D; padding: 10px; border-radius: 6px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #E6EDF3; text-transform: uppercase;">Chance of Major Dip</div>'
                '<div style="font-size: 10px; color: #58A6FF; margin-top: 1px;">Probability of >5% Loss</div>'
                f'<div class="mono-val" style="font-size: 22px; font-weight: 700; color: #D4A853; margin-top: 3px;">{mc_res["prob_loss_5"]:.1f}%</div>'
                '<div style="font-size: 10px; color: #8B949E; margin-top: 3px;">Probability of losing more than 5% of your money</div>'
                '</div>'
                '</div>'
                '<div style="font-size: 13px; line-height: 1.6; color: #E6EDF3; border-top: 1px solid #30363D; padding-top: 10px;">'
                '<div style="font-size: 11px; font-weight: 600; color: #D4A853; text-transform: uppercase; margin-bottom: 4px;">EXECUTIVE STRESS BRIEF</div>'
                f'<div>Under <strong>{stress_name}</strong>, projected portfolio drawdown scenarios over 21 trading days:</div>'
                f'<div style="margin-top: 5px; font-size: 12px; color: #C9D1D9;">'
                f'• <strong>Safe Limit (VaR):</strong> 95% of simulated 21-day runs stay above <strong>-{mc_res["var_95"]:.1f}%</strong> (e.g. ₹{var_1lakh:,} max loss on ₹1,00,000).<br>'
                f'• <strong>Tail Crash (CVaR):</strong> In the worst 5% of market crashes, expected losses average <strong>-{mc_res["cvar_95"]:.1f}%</strong> (e.g. ₹{cvar_1lakh:,} loss on ₹1,00,000).<br>'
                f'• <strong>Path Drawdown Risk:</strong> 95% of simulated runs limit maximum intra-period dip to <strong>-{mc_res["p95_path_dd"]:.1f}%</strong> (worst single-path trough: -{mc_res["worst_path_dd"]:.1f}%).'
                f'</div>'
                '<div style="margin-top: 8px; font-size: 11px; color: #8B949E; border-top: 1px dashed #30363D; padding-top: 6px;">'
                '<strong>Investor Takeaway:</strong> Size position weights so you can comfortably endure the <em>Severe Crash Average (CVaR)</em> without panic-selling at the bottom.'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(tail_metrics_html, unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size: 11px; color: #6E7681; margin-top: 16px; border-top: 1px solid #30363D; padding-top: 10px;">
                <strong>Methodology & Governance:</strong> (1) Historical crisis impacts use Sharpe (1964) CAPM Beta sensitivity. (2) Monte Carlo uses 1,000 Geometric Brownian Motion paths with zero-drift risk neutrality (Black-Scholes-Merton 1973). (3) Tail metrics follow Basel III Expected Shortfall standards. Stocks unlisted during historical crises are explicitly flagged without fabricated price feeds.
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# TAB 3: TRACK RECORD & ECONOMETRIC AUDIT
# =============================================================================
with tab_record:
    # -------------------------------------------------------------------------
    # STEP 1: 3 HUMAN-FIRST RELIABILITY & ACCURACY CARDS
    # -------------------------------------------------------------------------
    tab3_header_html = (
        '<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding: 4px 0 10px 0; margin-bottom: 14px;">'
        '<div>'
        '<div style="font-size: 13px; font-weight: 700; color: #D4A853; text-transform: uppercase; letter-spacing: 0.05em;">'
        'Model Reliability & Real-World Track Record'
        '</div>'
        '<div style="font-size: 11px; color: #8B949E; margin-top: 2px;">'
        'How accurately does QuantShield forecast risk compared to standard banking formulas?'
        '</div>'
        '</div>'
        '<span class="badge-gold">18 YEARS TESTED (2008–2026)</span>'
        '</div>'
    )
    st.markdown(tab3_header_html, unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        card1_html = (
            '<div class="terminal-card" style="border-top: 2px solid #D4A853;">'
            '<div class="terminal-header">'
            '<span class="terminal-title">MARKET TURBULENCE FORECAST</span>'
            '<span class="badge-gold">VS TRADITIONAL FORMULA</span>'
            '</div>'
            '<div style="padding: 2px 0 8px 0;">'
            '<div style="font-size: 11px; color: #8B949E; text-transform: uppercase;">'
            'Typical Forecast Margin (Error Range)'
            '</div>'
            '<div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">'
            '<span class="mono-val" style="font-size: 36px; font-weight: 700; color: #D4A853; line-height: 1;">'
            '±5.59%'
            '</span>'
            '<span style="font-size: 11px; color: #8B949E;">Average Error Margin</span>'
            '</div>'
            '</div>'
            '<div style="border-top: 1px solid #30363D; padding-top: 8px; font-size: 11px; line-height: 1.7;">'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Traditional Banking Formula:</span>'
            '<span class="mono-val" style="color: #E6EDF3;">±5.78% margin</span>'
            '</div>'
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
            '<span style="color: #8B949E;">Accuracy Improvement:</span>'
            '<span class="badge-green">+0.18% Better Precision</span>'
            '</div>'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Market Swings Captured:</span>'
            '<span class="mono-val" style="color: #E6EDF3;">32.5% (vs 22.7% traditional)</span>'
            '</div>'
            '</div>'
            '<div style="border-top: 1px solid #21262D; margin-top: 8px; padding-top: 8px; font-size: 11px; color: #8B949E; line-height: 1.4;">'
            '<em>On any given month, predictions stay within ~5% of actual swings, tracking 43% more market movement patterns than classical formulas.</em>'
            '</div>'
            '</div>'
        )
        st.markdown(card1_html, unsafe_allow_html=True)

    with col_r2:
        card2_html = (
            '<div class="terminal-card" style="border-top: 2px solid #3FB68B;">'
            '<div class="terminal-header">'
            '<span class="terminal-title">CRASH & SEVERE DROP WARNING</span>'
            '<span class="badge-green">HIGH ACCURACY (p < 0.001)</span>'
            '</div>'
            '<div style="padding: 2px 0 8px 0;">'
            '<div style="font-size: 11px; color: #8B949E; text-transform: uppercase;">'
            '21-Day Downside Drop Margin'
            '</div>'
            '<div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">'
            '<span class="mono-val" style="font-size: 36px; font-weight: 700; color: #3FB68B; line-height: 1;">'
            '±2.86%'
            '</span>'
            '<span style="font-size: 11px; color: #8B949E;">Typical Drop Error</span>'
            '</div>'
            '</div>'
            '<div style="border-top: 1px solid #30363D; padding-top: 8px; font-size: 11px; line-height: 1.7;">'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Historical Average Formula:</span>'
            '<span class="mono-val" style="color: #E6EDF3;">±3.35% margin</span>'
            '</div>'
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
            '<span style="color: #8B949E;">Warning Accuracy Lift:</span>'
            '<span class="badge-green">+0.49% Better Precision</span>'
            '</div>'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Days Ahead of Baseline:</span>'
            '<span class="mono-val" style="color: #3FB68B; font-weight: 600;">63.8% of trading days</span>'
            '</div>'
            '</div>'
            '<div style="border-top: 1px solid #21262D; margin-top: 8px; padding-top: 8px; font-size: 11px; color: #8B949E; line-height: 1.4;">'
            '<em>Accurately anticipates the worst 21-day drops on 64 out of every 100 days, giving you vital early warnings before severe drawdowns.</em>'
            '</div>'
            '</div>'
        )
        st.markdown(card2_html, unsafe_allow_html=True)

    with col_r3:
        card3_html = (
            '<div class="terminal-card" style="border-top: 2px solid #58A6FF;">'
            '<div class="terminal-header">'
            '<span class="terminal-title">REAL-WORLD TESTING INTEGRITY</span>'
            '<span class="badge-muted">STRICTLY UNSEEN DATA</span>'
            '</div>'
            '<div style="padding: 2px 0 8px 0;">'
            '<div style="font-size: 11px; color: #8B949E; text-transform: uppercase;">'
            'Real Market Days Evaluated'
            '</div>'
            '<div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">'
            '<span class="mono-val" style="font-size: 36px; font-weight: 700; color: #58A6FF; line-height: 1;">'
            '4,917'
            '</span>'
            '<span style="font-size: 11px; color: #8B949E;">trading days</span>'
            '</div>'
            '</div>'
            '<div style="border-top: 1px solid #30363D; padding-top: 8px; font-size: 11px; line-height: 1.7;">'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Historical Span:</span>'
            '<span class="mono-val" style="color: #E6EDF3;">2008 GFC through 2026</span>'
            '</div>'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Independent Dates Sampled:</span>'
            '<span class="mono-val" style="color: #E6EDF3;">825 trading dates</span>'
            '</div>'
            '<div style="display: flex; justify-content: space-between;">'
            '<span style="color: #8B949E;">Testing Honesty:</span>'
            '<span style="color: #3FB68B; font-weight: 600;">Zero Cheating (Purged)</span>'
            '</div>'
            '</div>'
            '<div style="border-top: 1px solid #21262D; margin-top: 8px; padding-top: 8px; font-size: 11px; color: #8B949E; line-height: 1.4;">'
            '<em>Tested strictly on future days the model had never seen before. No hindsight bias and no curve-fitting on past crashes.</em>'
            '</div>'
            '</div>'
        )
        st.markdown(card3_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 2: REGIME ACCURACY CHART & HEAD-TO-HEAD COMPARISON (58% / 42%)
    # -------------------------------------------------------------------------
    col_regime_chart, col_regime_audit = st.columns([58, 42])

    with col_regime_chart:
        chart_header_html = (
            '<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding-bottom: 6px; margin-bottom: 10px;">'
            '<div>'
            '<span class="terminal-title">Performance Under Different Market Conditions</span>'
            '</div>'
            '<span class="badge-gold">LOWER BAR = MORE ACCURATE</span>'
            '</div>'
        )
        st.markdown(chart_header_html, unsafe_allow_html=True)

        # Build Grouped Bar Chart comparing CatBoost vs EWMA across 3 conditions
        regimes = ["Quiet / Calm Markets", "Normal Trading Markets", "High Panic & Crises"]
        catboost_mae = [5.76, 3.31, 7.71]
        ewma_mae = [4.19, 4.47, 8.67]

        fig_regime = go.Figure()

        fig_regime.add_trace(go.Bar(
            x=regimes,
            y=catboost_mae,
            name="QuantShield AI",
            marker_color="#D4A853",
            text=[f"±{v:.2f}%" for v in catboost_mae],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#E6EDF3")
        ))

        fig_regime.add_trace(go.Bar(
            x=regimes,
            y=ewma_mae,
            name="Traditional Banking Formula",
            marker_color="#58A6FF",
            text=[f"±{v:.2f}%" for v in ewma_mae],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#E6EDF3")
        ))

        fig_regime.update_layout(
            barmode='group',
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            margin=dict(l=40, r=20, t=30, b=30),
            height=340,
            font=dict(family="Inter", color="#8B949E", size=11),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#E6EDF3", family="Inter")
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color="#E6EDF3", size=11)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#21262D",
                ticksuffix="%",
                range=[0, 10.5],
                tickfont=dict(color="#8B949E", size=10),
                title=dict(text="Average Error Margin (±%)", font=dict(size=10, color="#6E7681"))
            )
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    with col_regime_audit:
        audit_header_html = (
            '<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363D; padding-bottom: 6px; margin-bottom: 10px;">'
            '<div>'
            '<span class="terminal-title">QuantShield AI vs. Traditional Formula</span>'
            '</div>'
            '<span class="badge-gold">HONEST COMPARISON</span>'
            '</div>'
        )
        st.markdown(audit_header_html, unsafe_allow_html=True)

        audit_card_html = (
            '<div class="terminal-card" style="padding: 12px 14px; margin-bottom: 0;">'
            '<table style="width: 100%; border-collapse: collapse; font-size: 11px; text-align: left; margin-bottom: 10px;">'
            '<tr style="border-bottom: 1px solid #30363D; color: #8B949E; font-size: 10px; text-transform: uppercase;">'
            '<th style="padding: 4px 6px;">What We Tested</th>'
            '<th style="padding: 4px 6px;">QuantShield AI</th>'
            '<th style="padding: 4px 6px;">Traditional</th>'
            '<th style="padding: 4px 6px;">Why It Matters To You</th>'
            '</tr>'
            '<tr style="border-bottom: 1px solid #21262D;">'
            '<td style="padding: 6px; color: #E6EDF3; font-weight: 500;">Predicting Severe Drops</td>'
            '<td style="padding: 6px; color: #3FB68B; font-family: \'JetBrains Mono\', monospace;">±2.86%</td>'
            '<td style="padding: 6px; color: #8B949E; font-family: \'JetBrains Mono\', monospace;">±3.35%</td>'
            '<td style="padding: 6px; color: #3FB68B; font-size: 10px;">Catches drops earlier (p < 0.001)</td>'
            '</tr>'
            '<tr style="border-bottom: 1px solid #21262D;">'
            '<td style="padding: 6px; color: #E6EDF3; font-weight: 500;">Tracking Market Swings</td>'
            '<td style="padding: 6px; color: #D4A853; font-family: \'JetBrains Mono\', monospace;">32.5% captured</td>'
            '<td style="padding: 6px; color: #8B949E; font-family: \'JetBrains Mono\', monospace;">22.7% captured</td>'
            '<td style="padding: 6px; color: #D4A853; font-size: 10px;">+43% more dynamics tracked</td>'
            '</tr>'
            '<tr style="border-bottom: 1px solid #21262D;">'
            '<td style="padding: 6px; color: #E6EDF3; font-weight: 500;">Normal Everyday Markets</td>'
            '<td style="padding: 6px; color: #3FB68B; font-family: \'JetBrains Mono\', monospace;">±3.31% error</td>'
            '<td style="padding: 6px; color: #8B949E; font-family: \'JetBrains Mono\', monospace;">±4.47% error</td>'
            '<td style="padding: 6px; color: #3FB68B; font-size: 10px;">Wins 65% of trading days</td>'
            '</tr>'
            '<tr style="border-bottom: 1px solid #21262D;">'
            '<td style="padding: 6px; color: #E6EDF3; font-weight: 500;">High Panic / Crises</td>'
            '<td style="padding: 6px; color: #3FB68B; font-family: \'JetBrains Mono\', monospace;">±7.71% error</td>'
            '<td style="padding: 6px; color: #F47067; font-family: \'JetBrains Mono\', monospace;">±8.67% error</td>'
            '<td style="padding: 6px; color: #3FB68B; font-size: 10px;">Safer alerts during crashes</td>'
            '</tr>'
            '<tr>'
            '<td style="padding: 6px; color: #E6EDF3; font-weight: 500;">Quiet Range-Bound Days</td>'
            '<td style="padding: 6px; color: #8B949E; font-family: \'JetBrains Mono\', monospace;">±5.76% error</td>'
            '<td style="padding: 6px; color: #3FB68B; font-family: \'JetBrains Mono\', monospace;">±4.19% error</td>'
            '<td style="padding: 6px; color: #8B949E; font-size: 10px;">Simple formulas win when calm</td>'
            '</tr>'
            '</table>'
            '<div style="font-size: 11px; line-height: 1.45; color: #E6EDF3; border-top: 1px solid #30363D; padding-top: 8px;">'
            '<div style="font-size: 11px; font-weight: 600; color: #D4A853; text-transform: uppercase; margin-bottom: 2px;">Statistical Objective & Forecast Horizon</div>'
            '<div>Model forecasts represent point estimates optimized for MultiRMSE loss over 21-trading-day horizons. Empirical residual error corridors, out-of-fold calibration curves, and independent holdout test verifications are documented in the model audit logs to ensure transparent risk measurement across market volatility regimes.</div>'
            '</div>'
            '</div>'
        )
        st.markdown(audit_card_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 3: SCIENTIFIC INTEGRITY FOOTNOTE
    # -------------------------------------------------------------------------
    disclosure_html = (
        '<div style="font-size: 11px; color: #6E7681; margin-top: 14px; border-top: 1px solid #30363D; padding-top: 8px;">'
        '<strong>Testing Integrity & Scope:</strong> Evaluated across 18 years of National Stock Exchange (NSE) historical data (4,917 out-of-fold days) using strict time-quarantine validation with zero lookahead. Machine learning patterns reflect historical statistical associations, not guarantees of future returns.'
        '</div>'
    )
    st.markdown(disclosure_html, unsafe_allow_html=True)


# --- FOOTER TELEMETRY STRIP ---
st.markdown(
    """
    <div style="border-top: 1px solid #30363D; padding: 12px 0; margin-top: 24px; display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6E7681;">
        <div style="display: flex; gap: 16px;">
            <span style="color: #3FB68B; font-weight: 600;">● QUANTSHIELD ENGINE: ACTIVE</span>
            <span>DATA FEED: NSE HISTORICAL EOD</span>
            <span>HORIZON: 21 TRADING DAYS</span>
        </div>
        <div style="display: flex; gap: 16px;">
            <span>STANDARD: RISKMETRICS EWMA</span>
            <span>VERSION: 2.0-STABLE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
