import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Tuple

st.set_page_config(page_title="Portfolio Scatter App", layout="wide")
st.title("Portfolio Scatter: 5-Stock Input → Live Matplotlib Plot")

# =============================
# 🔧 Replace this with your function
# =============================
# Your function should accept two lists: tickers (List[str]) and weights (List[float])
# and return a matplotlib.figure.Figure to be displayed on the right.
#
# Example signature:
# def build_portfolio_scatter(tickers: List[str], weights: List[float]) -> plt.Figure:
#     ...
#     return fig
#
# For now, we include a placeholder so the app runs. Replace the function body
# with your actual implementation.

def build_portfolio_scatter(tickers: List[str], weights: List[float]) -> plt.Figure:
    fig, ax = plt.subplots()
    # Placeholder demo points
    xs = [0.05, 0.10, 0.20, 0.30]
    ys = [0.02, 0.06, 0.12, 0.18]
    ax.scatter(xs, ys)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected Return (E[R])")
    ax.set_title("Portfolio Scatter (placeholder) – Replace with your function output")
    ax.grid(True, alpha=0.3)
    return fig

# =============================
# UI Layout
# =============================
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Inputs (up to 5 stocks)")
    tickers: List[str] = []
    weights: List[float] = []

    for i in range(5):
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            t = c1.text_input(f"Ticker {i+1}", key=f"ticker_{i}").strip().upper()
            w = c2.number_input(
                f"Weight {i+1}", min_value=0.0, max_value=10.0, value=0.0, step=0.01, key=f"weight_{i}"
            )
            tickers.append(t)
            weights.append(float(w))

    # Clean: remove empty ticker rows but keep weight alignment for your function as needed
    # (If you prefer to pass zeros for empty tickers, comment out this filter block.)
    filtered: List[Tuple[str, float]] = [(t, w) for t, w in zip(tickers, weights) if t != ""]
    tickers = [t for t, _ in filtered]
    weights = [w for _, w in filtered]

    # Normalize only if sum > 1
    total_w = sum(weights)
    if total_w > 1.0:
        weights = [w / total_w for w in weights]
        st.info(
            f"Weights normalized because sum was {total_w:.4f} > 1. New sum = {sum(weights):.4f}.",
            icon="ℹ️",
        )

    st.caption(
        "If the sum of weights exceeds 1, they are automatically divided by the sum so the total ≤ 1."
    )

with right:
    st.subheader("Portfolio Scatter")

    # Guardrail: if no tickers provided, show a friendly prompt
    if len(tickers) == 0:
        st.write(
            "Enter at least one ticker on the left (with any non-zero weight) to render the chart."
        )
        # Show placeholder figure so layout stays stable
        fig, ax = plt.subplots()
        ax.set_xlabel("Volatility (σ)")
        ax.set_ylabel("Expected Return (E[R])")
        ax.set_title("Waiting for input…")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    else:
        # Call YOUR function here. It should return a matplotlib Figure.
        fig = build_portfolio_scatter(tickers, weights)
        st.pyplot(fig, use_container_width=True)

# ==============
# Footer details
# ==============
st.markdown(
    """
    <div style='opacity:0.7; font-size: 0.9em;'>
    <strong>Notes</strong>:
    <ul>
      <li>Provide 1–5 tickers with associated non-negative weights.</li>
      <li>Weights are auto-normalized only if their sum exceeds 1.</li>
      <li>Replace <code>build_portfolio_scatter</code> with your implementation that returns a Matplotlib Figure.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
