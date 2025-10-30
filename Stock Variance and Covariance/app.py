
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math
import pandas_datareader.data as web
import random 

class riskFreeAsset:
    def __init__(self, rate):
        self.ticker = "Treasury Bond"
        self.rate = rate  # annual risk-free rate
        self.DR = self.get_daily_returns()
        self.ER = self.get_expected_return()
        self.VAR = self.get_variance()
        self.STD = self.get_standard_deviation()

    def get_expected_return(self):
        return self.rate / 252  # daily return if needed

    def get_variance(self):
        return 0

    def get_standard_deviation(self):
        return 0

    def get_annualized_volatility(self):
        return 0
    
    def get_daily_returns(self):
        daily = []
        for x in range(252):
            daily.append(self.rate / 252)

        return daily

class stock:

    def __init__(self,ticker):
        self.ticker = ticker
        self.closes = self.get_historical_closes()
        self.DR = self.get_daily_returns()
        self.ER = self.get_expected_return()
        self.VAR = self.get_variance()
        self.STD = self.get_standard_deviation()
        
        
    def get_historical_closes(self):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)

        data = web.DataReader(self.ticker, "stooq", start_date, end_date)
        data = data.sort_index()

        daily_closes = data['Close']
        closes_list = daily_closes.tolist()
        
        return closes_list
    
    #past year
    def get_daily_returns(self):
        closes = self.closes
        return [(closes[i]-closes[i-1]) / closes[i-1] for i in range(1,len(closes))]

    def get_expected_return(self): 
        returns = self.DR
        return sum(returns) / (len(returns) - 1)
        
    def get_variance(self):
        returns = self.DR
        expected = self.ER
        summation = 0

        for R in returns:
            summation += (R - expected) ** 2

        return summation/ (len(returns) - 1)
    
    def get_covariance(stockA,stockB):
        returnsA = stockA.DR
        expectedA = stockA.ER
        returnsB = stockB.DR
        expectedB = stockB.ER
        summation = 0

        for RA, RB in zip(returnsA,returnsB):
            summation += (RA-expectedA) * (RB-expectedB)

        return summation / (len(returnsA)-1)

    def get_standard_deviation(self):
        return math.sqrt(self.VAR)

    def get_annualized_volatility(self):
        std = self.STD
        return std * math.sqrt(252)

    def get_correlation(stockA, stockB):
        stdA = stockA.STD
        stdB = stockB.STD
        cov = stock.get_covariance(stockA,stockB)

        return cov / (stdA * stdB)

class portfolio:
    
    def __init__(self,riskFreeRate,*args):
        self.stocks = []
        self.weights = []
        for arg in args:
            self.stocks.append(arg[0])
            self.weights.append(arg[1])
        
        self.riskFreeRate = riskFreeRate
        self.ER = self.get_expected_return()
        self.STD = self.get_standard_deviation()

    def get_expected_return(self):
        summation = 0
        for stock,weight in zip(self.stocks,self.weights):
            summation += stock.ER * weight

        return summation

    def get_variance(self):
        summation = 0

        for s,weight in zip(self.stocks,self.weights):
            summation += (weight ** 2) * s.VAR

        for i in range(len(self.stocks)):
            for j in range(len(self.stocks)):
                if i != j and i > j:
                    summation += 2 * stock.get_covariance(self.stocks[i],self.stocks[j]) * self.weights[i] * self.weights[j]

        return summation
    
    def get_standard_deviation(self):
        return math.sqrt(self.get_variance())
    
    def get_yearly_volatility(self):
        return self.get_standard_deviation() * math.sqrt(252)
    
    def get_sharpe_ratio(self):
        return (self.ER- self.riskFreeRate) / self.STD
    
    def get_point(self):
        return (self.STD, self.ER)
    
    def get_weights(self):
        return self.weights
    
    def get_display(self):
        tuples = zip(self.stocks,self.weights)
        disp = ""
        for t in tuples:
            disp += f"<br>{t[0].ticker} : {round(t[1],2)}"
        return disp

class portfolioAnalyzer:
    def __init__(self,riskFreeRate,includeBond,*args):
        self.stocks = []
        for arg in args:
            self.stocks.append(arg)
        if includeBond:
            self.stocks.append(riskFreeAsset(riskFreeRate))
        self.riskFreeRate = riskFreeRate

    def get_clean_points(self,numPoints,allowShort=False,percentile=90):
        points = self.get_plot(numPoints,allowShort)
        # Step 1: remove all with negative y
        filtered = [(t,p) for t,p in points if t[1] >= 0]
        if not filtered:
            return []
        scores = np.array([np.sqrt(t[0]**2 + t[1]**2) for t,p in filtered])
        cutoff = np.percentile(scores, percentile)
        cleaned = [t for t, s in zip(filtered, scores) if s <= cutoff]
        
        maxX = max(pair[0][0] for pair in points)
        maxY = max(pair[0][1] for pair in points)

        return cleaned, maxX, maxY

    def get_plot(self,a,allowShort=False):
        ps = self.get_x_random_portfolios(a,allowShort)
        tuples = []
        for p in ps:
            tuples.append((p.get_point(),p))
        return tuples
    
    def get_x_random_portfolios(self,x,allowShort=False):
        portfolios = []
        for i in range(x):
            p = portfolio(self.riskFreeRate,*list(zip(self.stocks,portfolioAnalyzer.get_random_weights(len(self.stocks),allowShort))))
            portfolios.append(p)
        return portfolios

    def get_random_weights(n, allow_short=False):
        if allow_short:
            # Generate random weights in [-1, 2]
            weights = [random.uniform(-1, 2) for _ in range(n)]
        else:
            # Generate random weights in [0, 1]
            weights = [random.random() for _ in range(n)]
        
        # Avoid division by zero (all weights very close to 0)
        while abs(sum(weights)) < 1e-8:
            if allow_short:
                weights = [random.uniform(-1, 2) for _ in range(n)]
            else:
                weights = [random.random() for _ in range(n)]
        
        # Normalize so they sum to 1
        total = sum(weights)
        normalized_weights = [w / total for w in weights]
        
        return normalized_weights
    
class FrontierVisualizer:
    
    def __init__(self,riskFreeRate,stocks,weights,includeBond=False,allowShort=False,numPoints=5000,percentileCutoff=99):
        self.r = riskFreeRate
        self.includeBone = includeBond
        self.allowShort = allowShort
        self.numPoints = numPoints
        self.percentileCutoff = percentileCutoff
        self.stocks = stocks
        self.userP = portfolio(self.r,*zip(stocks,weights))

        p = portfolioAnalyzer(self.r,includeBond,*stocks)
        self.points,self.maxX,self.maxY = p.get_clean_points(self.numPoints,self.allowShort,self.percentileCutoff)
        #self.optimal_p = self.get_optimal_portfolio(self.points)
    
    def get_ticker_list(self):
        return [s.ticker for s in self.stocks]

    def get_optimal_portfolio(self,points):
        bestSharpe = (-100,None)
        for c,p in points:
            c_sharpe = self.get_sharpe_from_point(c)
            if c_sharpe > bestSharpe[0]:
                bestSharpe = (c_sharpe,p)
        if bestSharpe[1] == None:
            return None
        return (bestSharpe[1].get_point(), bestSharpe[1])

    def get_sharpe_from_point(self,c):
        return (c[1]-self.r)/c[0]
    
    def update_user_point(self,weights):
        self.userP = portfolio(self.r,*zip(self.stocks,weights))

    def get_plot(self):
        """
        Build a Plotly figure for Streamlit's st.plotly_chart(fig).
        Supports hover tooltips that display the extra value 'v' for each point.
        """

        #optimal_point = self.optimal_p
        user_point = self.userP.get_point()
        user_label = "Your portfolio"
        #opt_label = f"Optimal (approx): {optimal_point[1].get_display()}"
        pts = list(self.points)
        if not pts:
            fig = go.Figure()
            fig.update_layout(
                title="Portfolio Risk–Return Scatter",
                xaxis_title="Portfolio Std Dev, σ(Rₚ)",
                yaxis_title="Expected Return, E[Rₚ]",
            )
            return fig

        X = np.array([p[0][0] for p in pts], dtype=float)
        Y = np.array([p[0][1] for p in pts], dtype=float)
        V = [p[1] for p in pts]  # keep as-is for hover text

        # Main cloud
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=X, y=Y,
            mode="markers",
            marker=dict(size=5),
            name="Portfolios",
            text=[f"Weights: {v.get_display()}" for v in V],
            hovertemplate="σ(Rₚ)=%{x:.4f}<br>E[Rₚ]=%{y:.4f}<br>%{text}<extra></extra>"
        ))

        # Highlights
        fig.add_trace(go.Scatter(
            x=[user_point[0]], y=[user_point[1]],
            mode="markers",
            marker=dict(size=10, symbol="x", line=dict(width=1.5, color="white")),
            name=user_label,
            text=[user_label],
            textposition="top left",
            hovertemplate="σ(Rₚ)=%{x:.4f}<br>E[Rₚ]=%{y:.4f}<extra></extra>"
        ))
        fig.add_annotation(
            x=user_point[0], y=user_point[1],
            text="✕", showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(214,39,40,0.9)",  # optional halo
            bordercolor="white", borderwidth=2
        )
        """
        if optimal_point != None:
            fig.add_trace(go.Scatter(
                x=[optimal_point[0][0]], y=[optimal_point[0][1]],
                mode="markers",
                marker=dict(size=10, symbol="triangle-up", color="#d62728",
                            line=dict(width=1.5, color="white")),
                name=opt_label,
                text=[opt_label],
                textposition="bottom left",
                hovertemplate="σ(Rₚ)=%{x:.4f}<br>E[Rₚ]=%{y:.4f}<br>%{text}<extra></extra>"
            ))
            fig.add_annotation(
                x=optimal_point[0][0], y=optimal_point[0][1],
                text="▲", showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(214,39,40,0.9)",  # optional halo
                bordercolor="white", borderwidth=2
            )
            """
    
        fig.update_layout(
            title="Portfolio Risk–Return Scatter",
            xaxis_title="Portfolio Std Dev, σ(Rₚ)",
            yaxis_title="Expected Return, E[Rₚ)",
            legend=dict(borderwidth=0),
            hovermode="closest",
        )
        return fig






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





        