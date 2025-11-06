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
        `"""
        Returns a list of daily close (prefer adjusted if available) for the last 365 days.
        Tries Stooq via pandas_datareader first, then falls back to Yahoo via yfinance.
        """
        # Use end as "tomorrow" so most APIs include today's bar if available.
        end_date = datetime.today() + timedelta(days=1)
        start_date = end_date - timedelta(days=366)  # one extra day to be safe

        data = None
        errors = []

        # 1) Try Stooq via pandas_datareader
        try:
            data = web.DataReader(self.ticker, "stooq", start_date, end_date)
        except Exception as e:
            errors.append(f"stooq error: {e}")

        # 2) Fallback to Yahoo via yfinance if needed
        if data is None or data.empty:
            try:
                import yfinance as yf
                # yfinance returns columns: Open, High, Low, Close, Adj Close, Volume
                data = yf.download(self.ticker, start=start_date.date(), end=end_date.date(), progress=False)
            except Exception as e:
                errors.append(f"yfinance error: {e}")

        if data is None or data.empty:
            raise RuntimeError(
                f"No price data returned for ticker {self.ticker}. "
                f"Tried Stooq and Yahoo. Details: {' | '.join(errors)}"
            )

        # Ensure a proper DateTime index and ascending order
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        # Be resilient to column-casing and naming differences
        cols_lower = {c.lower(): c for c in data.columns}
        close_col = None

        # Prefer adjusted close when available
        for candidate in ("adj close", "adj_close", "adjusted close"):
            if candidate in cols_lower:
                close_col = cols_lower[candidate]
                break

        # Then try plain close
        if close_col is None:
            for candidate in ("close",):
                if candidate in cols_lower:
                    close_col = cols_lower[candidate]
                    break

        # As last resort, use last column if it looks numeric
        if close_col is None:
            if len(data.columns) >= 1:
                close_col = data.columns[-1]
            else:
                raise ValueError(
                    f"No close-like column found for {self.ticker}. "
                    f"Available columns: {list(data.columns)}"
                )

        # Clean and return as list
        daily_closes = (
            data[close_col]
            .dropna()
            .astype(float)
            .tolist()
        )

        if not daily_closes:
            raise RuntimeError(f"All close values were NaN/empty for {self.ticker} after cleaning.")

        return daily_closes

    
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

