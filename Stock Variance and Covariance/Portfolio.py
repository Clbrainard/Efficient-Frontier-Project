from Securities import stock,riskFreeAsset
import math

class portfolio:
    
    def __init__(self,riskFreeRate,*args):
        self.stocks = []
        self.weights = []
        for arg in args:
            self.stocks.append(arg[0])
            self.weights.append(arg[1])
        
        self.riskFreeRate = riskFreeRate

    def get_expected_return(self):
        summation = 0
        for stock,weight in zip(self.stocks,self.weights):
            summation += stock.get_expected_return() * weight

        return summation

    def get_variance(self):
        summation = 0

        for s,weight in zip(self.stocks,self.weights):
            summation += (weight ** 2) * s.get_variance()

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
        return (self.get_expected_return() - self.riskFreeRate) / self.get_standard_deviation()
    
    def get_point(self):
        return (self.get_standard_deviation(), self.get_expected_return())

