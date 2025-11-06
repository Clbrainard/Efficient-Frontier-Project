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

        # The data source may name the close column differently depending on the reader.
        # Try several common names, fall back to case-insensitive match, then last column.
        daily_closes = None
        for col in ("Close", "close", "Adj Close", "adj close", "Adj_Close", "adj_close"):
            if col in data.columns:
                daily_closes = data[col]
                break

        if daily_closes is None:
            # Try case-insensitive match
            cols_lower = {c.lower(): c for c in data.columns}
            if 'close' in cols_lower:
                daily_closes = data[cols_lower['close']]

        if daily_closes is None:
            # As a last resort, use the last column (often the close or adjusted close)
            if len(data.columns) >= 1:
                daily_closes = data.iloc[:, -1]
            else:
                raise ValueError(f"No close price column found for ticker {self.ticker}. Available columns: {list(data.columns)}")

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

