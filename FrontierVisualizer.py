from PortfolioAnalysis import portfolioAnalyzer
from typing import Iterable, Tuple, Any
import numpy as np
import plotly.graph_objects as go
from Portfolio import portfolio

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



                    



            
            

