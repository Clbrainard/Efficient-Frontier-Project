from Securities import stock,riskFreeAsset
import math
import numpy as np

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

