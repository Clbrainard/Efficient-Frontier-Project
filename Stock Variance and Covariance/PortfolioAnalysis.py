from Securities import riskFreeAsset,stock
from Portfolio import portfolio
import random 
import math
import matplotlib.pyplot as plt
import numpy as np


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
        filtered = [t for t,p in points if t[1] >= 0]
        if not filtered:
            return []
        scores = np.array([np.sqrt(t[0]**2 + t[1]**2) for t in filtered])
        cutoff = np.percentile(scores, percentile)
        cleaned = [t for t, s in zip(filtered, scores) if s <= cutoff]
        
        maxX = np.max(points[:, 0])
        maxY = np.max(points[:, 1])

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
    


