from Securities import riskFreeAsset,stock
from Portfolio import portfolio
from PortfolioAnalysis import portfolioAnalyzer
import random 
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

class frontierAnalyzer:

    def __init__(self,riskFreeRate,includeBond,*args):
        self.stocks=[]
        for arg in args:
            self.stocks.append(stock(arg))
        self.riskFreeRate = riskFreeRate
        self.analyzer = portfolioAnalyzer(riskFreeRate,includeBond,*self.stocks)

    def clean_points(points,percentile=90):
        """
        Cleans a list of (x, y, p) tuples by:
        1. Removing tuples with negative y.
        2. Removing the top 10% outliers (largest x and y combined).
        """
        # Step 1: remove all with negative y
        filtered = [t for t in points if t[1] >= 0]

        if not filtered:
            return []

        # Step 2: remove top 10% outliers by combined magnitude
        # Compute a "score" = sqrt(x^2 + y^2) for each point
        scores = np.array([np.sqrt(t[0]**2 + t[1]**2) for t in filtered])
        
        # Find cutoff for the top 10%
        cutoff = np.percentile(scores, percentile)

        # Keep only points below cutoff
        cleaned = [t for t, s in zip(filtered, scores) if s <= cutoff]

        return cleaned
    
    def getMaxes(points):
        maxX = np.max(points[:, 0])
        maxY = np.max(points[:, 1])
        return maxX,maxY

    def plot_points(points, max_x, max_y, log_scale=False):

        # Unpack main points (list of (x, y, portfolio) tuples)
        x_vals, y_vals, portfolios = zip(*points)
        x_vals = np.array(x_vals, dtype=float)
        y_vals = np.array(y_vals, dtype=float)

        # Choose small positive epsilons (so log scales won't break and all points are in Q1)
        epsx = max(1e-12, 1e-4 * max_x)
        epsy = max(1e-12, 1e-4 * max_y)

        # Nudge any non-positive main points into Q1
        x_plot = np.where(x_vals <= 0, epsx, x_vals)
        y_plot = np.where(y_vals <= 0, epsy, y_vals)

        plt.figure(figsize=(6, 6))
        plt.scatter(x_plot, y_plot, color='blue', marker='o', s=10, label='Points')

        

        # Axis limits strictly positive (top-right quadrant only)
        # Use small padding but never below eps
        x_lo = max(epsx, 0.8 * max(epsx, np.min(x_plot)))
        y_lo = max(epsy, 0.8 * max(epsy, np.min(y_plot)))
        plt.xlim(0, max_x)
        plt.ylim(0, max_y)

        # Apply scaling
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            plt.title('Scatter Plot (Log Scale)')
        else:
            plt.title('Scatter Plot (Linear Scale)')

        plt.xlabel('Standard deviation of returns' + (' (log scale)' if log_scale else ''))
        plt.ylabel('Expected Returns' + (' (log scale)' if log_scale else ''))
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

    def compute_sharpe(riskFree,std,ER):
        return (ER-riskFree) / std
    
    def find_optimal_sharpe(riskFree,points):
        bestSharpe = 0
        bestPoint = None
        for p in points:
            newSharpe = frontierAnalyzer.compute_sharpe(riskFree,p[0],p[1])
            if newSharpe > bestSharpe:
                bestSharpe = newSharpe
                bestPoint = p
        return bestPoint


"""  
high_corr_high_vol = [
    stock("NVDA"),  # NVIDIA - high beta, momentum-driven
    stock("AMD"),   # Advanced Micro Devices
    stock("TSLA"),  # Tesla - volatile growth stock
    stock("SMCI"),  # Super Micro Computer - volatile AI exposure
    stock("META"),  # Meta Platforms - correlated to tech momentum
]

high_corr_low_vol = [
    stock("KO"),   # Coca-Cola
    stock("PG"),   # Procter & Gamble
    stock("PEP"),  # PepsiCo
    stock("JNJ"),  # Johnson & Johnson
    stock("WMT"),  # Walmart
]
"""

low_corr_high_vol = [
    stock("PLTR"),  # Palantir - software volatility
    stock("XOM"),   # ExxonMobil - energy cycle exposure
    stock("COIN"),  # Coinbase - crypto correlation
    stock("RIVN"),  # Rivian - EV volatility
    stock("NVCR"),  # NovoCure - biotech swings
]

"""
low_corr_low_vol = [
    stock("NEE"),   # NextEra Energy - utilities
    stock("PG"),    # Consumer staples
    stock("MSFT"),  # Large-cap tech stability
    stock("MMM"), # Diversified holding company
    stock("BMY"),   # Bristol Myers Squibb - healthcare
]
"""
riskfree = 0.04

pO = portfolioAnalyzer(riskfree,False,*low_corr_high_vol)
points = pO.get_plot(5000,True)

points = frontierAnalyzer.clean_points(points,100)
maxX,maxY = frontierAnalyzer.getMaxes(np.array(points))
optimal = frontierAnalyzer.find_optimal_sharpe(riskfree,points)


frontierAnalyzer.plot_points(points,0.25,0.04)


