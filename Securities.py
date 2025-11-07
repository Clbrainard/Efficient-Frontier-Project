import yfinance as yf
from datetime import datetime, timedelta
import math
import pandas as pd
import requests
import streamlit as st

api_key=st.secrets["KEY"]



class riskFreeAsset:
    def __init__(self, rate):
        self.ticker = "Treasury Bond"
        self.rate = rate 
        self.DR = self.get_daily_returns()
        self.ER = self.get_expected_return()
        self.VAR = self.get_variance()
        self.STD = self.get_standard_deviation()

    def get_expected_return(self):
        return self.rate / 252 

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

    def __init__(self,ticker,key):
        self.key = key
        self.ticker = ticker
        self.closes = self.get_historical_closes()
        self.DR = self.get_daily_returns()
        self.ER = self.get_expected_return()
        self.VAR = self.get_variance()
        self.STD = self.get_standard_deviation()
        
    
    def get_historical_closes(self):
        end = datetime.now()
        start = end - timedelta(days=365)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        url = f"https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{start_str}/{end_str}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.key}

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("results", [])

        if not data:
            raise ValueError(f"No data found for ticker '{self.ticker}'")

        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)

        return df["c"]  # 'c' is the close price

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
