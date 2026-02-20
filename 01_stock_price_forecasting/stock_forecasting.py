"""
Stock Price Forecasting and Time Series Analysis
=================================================
This script performs comprehensive time series analysis and forecasting on FAANG stocks
using ARIMA models and various statistical techniques.

Author: PREETHI R
Project: Finance Analytics Portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)


def download_stock_data(tickers, start_date='2010-01-01'):
    """
    Download historical stock data from Yahoo Finance
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data
    
    Returns:
        DataFrame with adjusted close prices
    """
    print(f"Downloading data for {', '.join(tickers)}...")
    data = yf.download(tickers, start=start_date)['Adj Close']
    data.ffill(inplace=True)
    print(f"Downloaded {len(data)} records")
    return data


def plot_price_trends(data, tickers):
    """Plot adjusted close prices for all stocks"""
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(data[ticker], label=ticker, linewidth=2)
    plt.title('Adjusted Close Prices of FAANG Stocks', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig('price_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Price trends chart saved as 'price_trends.png'")


def calculate_and_plot_returns(data, tickers):
    """Calculate and visualize daily log returns"""
    returns = np.log(data / data.shift(1))
    
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(returns[ticker], label=ticker, alpha=0.7)
    plt.title('Daily Log Returns of FAANG Stocks', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Log Return', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('daily_returns.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Daily returns chart saved as 'daily_returns.png'")
    
    return returns


def calculate_and_plot_volatility(returns, tickers, window=30):
    """Calculate and visualize rolling volatility"""
    volatility = returns.rolling(window=window).std()
    
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(volatility[ticker], label=ticker, linewidth=2)
    plt.title(f'{window}-Day Rolling Volatility of FAANG Stocks', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility (Std Dev)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig('volatility.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Volatility chart saved as 'volatility.png'")
    
    return volatility


def adf_test(series, name=''):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Args:
        series: Time series data
        name: Name of the series for display
    """
    result = adfuller(series.dropna())
    print(f'\n{"="*60}')
    print(f'ADF Test Results for {name}')
    print(f'{"="*60}')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print(f'\n✓ Result: Series is STATIONARY (reject null hypothesis)')
    else:
        print(f'\n✗ Result: Series is NON-STATIONARY (fail to reject null hypothesis)')
    print(f'{"="*60}\n')


def train_arima_model(data, ticker='AAPL', order=(5, 1, 0), train_size=0.8):
    """
    Train ARIMA model and make predictions
    
    Args:
        data: Stock price data
        ticker: Stock ticker symbol
        order: ARIMA order (p, d, q)
        train_size: Proportion of data for training
    
    Returns:
        train, test, predictions, model
    """
    print(f"\nTraining ARIMA{order} model for {ticker}...")
    
    series = data[ticker].dropna()
    split_idx = int(len(series) * train_size)
    train, test = series[:split_idx], series[split_idx:]
    
    # Fit model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    
    print(f"✓ Model trained successfully")
    print(f"  Training samples: {len(train)}")
    print(f"  Testing samples: {len(test)}")
    
    return train, test, predictions, model_fit


def evaluate_model(test, predictions):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    print(f'\n{"="*60}')
    print(f'Model Evaluation Metrics')
    print(f'{"="*60}')
    print(f'Mean Absolute Error (MAE):  ${mae:.2f}')
    print(f'Root Mean Squared Error (RMSE):  ${rmse:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE):  {mape:.2f}%')
    print(f'{"="*60}\n')
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def plot_predictions(train, test, predictions, ticker='AAPL'):
    """Plot actual vs predicted prices"""
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data', color='blue', linewidth=2)
    plt.plot(test.index, test, label='Actual Prices', color='green', linewidth=2)
    plt.plot(test.index, predictions, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
    plt.title(f'{ticker} Stock Price Prediction using ARIMA', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{ticker}_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Prediction chart saved as '{ticker}_prediction.png'")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("STOCK PRICE FORECASTING AND TIME SERIES ANALYSIS")
    print("="*80 + "\n")
    
    # Define stock tickers
    tickers = ['AAPL', 'AMZN', 'NFLX', 'GOOGL', 'META']
    
    # Step 1: Download data
    data = download_stock_data(tickers)
    
    # Step 2: Plot price trends
    plot_price_trends(data, tickers)
    
    # Step 3: Calculate and plot returns
    returns = calculate_and_plot_returns(data, tickers)
    
    # Step 4: Calculate and plot volatility
    volatility = calculate_and_plot_volatility(returns, tickers)
    
    # Step 5: Stationarity test
    adf_test(data['AAPL'], 'AAPL Price Series')
    
    # Step 6: Train ARIMA model
    train, test, predictions, model = train_arima_model(data, ticker='AAPL')
    
    # Step 7: Evaluate model
    metrics = evaluate_model(test, predictions)
    
    # Step 8: Plot predictions
    plot_predictions(train, test, predictions, ticker='AAPL')
    
    # Summary statistics
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY STATISTICS")
    print("="*80)
    print(f"\nAnnualized Returns:")
    annual_returns = returns.mean() * 252
    for ticker in tickers:
        print(f"  {ticker}: {annual_returns[ticker]:.2%}")
    
    print(f"\nAnnualized Volatility:")
    annual_vol = returns.std() * np.sqrt(252)
    for ticker in tickers:
        print(f"  {ticker}: {annual_vol[ticker]:.2%}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    print("📊 Generated Files:")
    print("  - price_trends.png")
    print("  - daily_returns.png")
    print("  - volatility.png")
    print("  - AAPL_prediction.png")
    print("\n✅ All analysis completed successfully!")


if __name__ == "__main__":
    main()
