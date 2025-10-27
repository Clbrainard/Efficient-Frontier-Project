from Portfolio import portfolio
from FrontierVisualizer import FrontierVisualizer
import matplotlib.pyplot as plt
from Securities import stock

r=0.04
stocks = ["AAPL","INTC","PLTR"]
stocks = [stock(t) for t in stocks]
weights = [0.3,0.3,0.4]

userP = portfolio(r,*zip(stocks,weights))
V = FrontierVisualizer(r,stocks)
plot = V.get_plot(userP.get_point())

plot.show() 