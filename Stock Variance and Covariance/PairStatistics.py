from Securities import stockStatistics
import math

class PairStatistics:
    def __init__(self, tickerA, tickerB):
        # store stock objects for later calculations
        self.stockA = stockStatistics(tickerA)
        self.stockB = stockStatistics(tickerB)

    def get_covariance(self):
        returnsA = self.stockA.get_daily_returns()
        returnsB = self.stockB.get_daily_returns()
        n = min(len(returnsA), len(returnsB))
        if n <= 1:
            return 0.0
        # align to the same length (use last n)
        returnsA = returnsA[-n:]
        returnsB = returnsB[-n:]
        meanA = sum(returnsA) / n
        meanB = sum(returnsB) / n
        cov = sum((a - meanA) * (b - meanB) for a, b in zip(returnsA, returnsB)) / (n - 1)
        return cov

    def get_correlation(self):
        cov = self.get_covariance()
        varA = self.stockA.get_variance()
        varB = self.stockB.get_variance()
        if varA <= 0 or varB <= 0:
            return 0.0
        return cov / math.sqrt(varA * varB)

if __name__ == "__main__":
    p = PairStatistics("AAPL", "AAPL")
    print("Covariance:", p.get_covariance())
    print("Correlation:", p.get_correlation())