import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

def run_bitcoin_forecast():
    print("Forecasting Bitcoin Prices using Exponential Smoothing...")
    os.makedirs('images', exist_ok=True)
    
    # Generate random walk data
    np.random.seed(42)
    prices = 40000 + np.cumsum(np.random.normal(50, 1000, 100))
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({'Date': dates, 'Price': prices}).set_index('Date')
    
    # Model
    model = ExponentialSmoothing(df['Price'], trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(20)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Historical Price')
    plt.plot(pd.date_range(start=dates[-1], periods=20), forecast, label='Forecast', color='orange')
    plt.title('Bitcoin Price Forecast (Next 20 Days)')
    plt.legend()
    plt.savefig('images/bitcoin_forecast.png')
    plt.close()
    print("Success: Bitcoin forecast saved to images.")

if __name__ == "__main__":
    run_bitcoin_forecast()
