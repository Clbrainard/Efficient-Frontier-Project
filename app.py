
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math

import random 
from Securities import stock
from FrontierVisualizer import FrontierVisualizer
from Portfolio import portfolio




st.set_page_config(page_title="Efficient Frontier Tool", layout="wide")

if "fig" not in st.session_state:
    st.session_state["fig"] = None
    st.session_state["pv"] = None
    st.session_state["weights"] = None
    st.session_state["bond"] = None

st.title("Efficient Frontier")
st.caption("Choose 2–5 stocks and weights in the sidebar, set the risk-free rate, then click **Render**.")

# ----- Sidebar -----
st.sidebar.header("Inputs")

# Risk-free rate
rf = st.sidebar.number_input(
    "Risk-free rate (decimal, e.g., 0.03 = 3%)",
    min_value=-0.50, max_value=0.50, step=0.001, value=0.03, format="%.4f"
)

# Number of pairs (2–5)
n_pairs = st.sidebar.slider("Number of stocks/weights", min_value=2, max_value=5, value=5)

# NEW: Number of points on the frontier (500–8000), default around n_pairs*1500
default_points = int(np.clip(n_pairs * 1500, 500, 8000))
num_points = st.sidebar.slider("Number of frontier points", min_value=500, max_value=8000, value=default_points, step=100)

# NEW: Include a bond (risk-free asset) in the frontier calculation
include_bond = st.sidebar.checkbox("Include bond (risk-free asset) in frontier calc", value=False)

# Defaults
default_tickers_pool = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
default_weights = [round(1.0 / n_pairs, 2)] * n_pairs

st.sidebar.subheader("Tickers & Weights")

tickers = []
weights = []

# Two columns to make the inputs compact
col_t, col_w = st.sidebar.columns([2, 1])

for i in range(n_pairs):
    default_ticker = default_tickers_pool[i] if i < len(default_tickers_pool) else ""
    t = col_t.text_input(f"Ticker {i+1}", value=default_ticker, key=f"ticker_{i}").strip().upper()
    w = col_w.number_input(
        f"Weight {i+1}",
        min_value=0.0, max_value=1.0, step=0.01,
        value=default_weights[i] if i < len(default_weights) else 0.0,
        format="%.4f",
        key=f"weight_{i}"
    )
    tickers.append(t)
    weights.append(w)

# Utilities
w_sum = float(np.sum(weights))
st.sidebar.markdown(f"**Current weight sum:** `{w_sum:.4f}`")

# Render button
render = st.sidebar.button("Render", type="primary")

#update_user = st.sidebar.button("Update user point only")

# ----- Main Area -----

# ----- Main Area -----

if render:
    # Basic validations
    if any(t == "" for t in tickers):
        st.error("Please fill in all tickers.")
        st.stop()

    w = np.array(weights, dtype=float)

    try:
        if st.session_state["pv"] != None and tickers == st.session_state["pv"].get_ticker_list() and st.session_state["bond"] == include_bond:
            st.session_state["weigths"] = weights
            st.session_state["bond"] = include_bond
            st.session_state["pv"].update_user_point(weights)
            st.session_state["fig"] = st.session_state["pv"].get_plot()
        else:
            stocks = [stock(t) for t in tickers]
            st.session_state["bond"] = include_bond
            obj = FrontierVisualizer(rf, stocks, w.tolist(), numPoints=int(num_points),includeBond=include_bond)
            st.session_state["weigths"] = weights
            st.session_state["pv"] = obj
            st.session_state["fig"] = st.session_state["pv"].get_plot()

  
        # Keep the object + baseline inputs so we can update the user point without recomputing the frontier
        st.session_state["base_inputs"] = {
            "rf": rf,
            "tickers": tickers.copy(),
            "num_points": int(num_points),
            "include_bond": include_bond,
}

        with st.expander("Inputs used"):
            st.write({
                "risk_free_rate": rf,
                "tickers": tickers,
                "weights": [float(x) for x in w],
                "num_points": int(num_points),
                "include_bond": include_bond,
            })

    except Exception as e:
        st.error(f"Failed to render efficient frontier: {e}")


if st.session_state.get("fig") is None:
        st.info("Adjust the number of pairs (2–5), fill in tickers and weights, set the risk-free rate, choose the number of frontier points, optionally include a bond, then click **Render**.")
else:
        st.subheader("Efficient Frontier")
        st.plotly_chart(st.session_state["fig"], use_container_width=True)





        
