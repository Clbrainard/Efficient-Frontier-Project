import yfinance as yf
from datetime import datetime, timedelta
import math
import pandas_datareader.data as web

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


