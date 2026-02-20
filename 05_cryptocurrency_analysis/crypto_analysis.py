"""
Cryptocurrency Price Analysis and Forecasting
==============================================
This script performs comprehensive cryptocurrency analysis including price forecasting,
volatility modeling, and Value at Risk (VaR) calculations.

Author: PREETHI R
Project: Finance Analytics Portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)


def download_crypto_data(tickers, start_date='2017-01-01'):
    """
    Download historical cryptocurrency data
    
    Args:
        tickers: List of crypto ticker symbols (e.g., ['BTC-USD', 'ETH-USD'])
        start_date: Start date for historical data
    
    Returns:
        DataFrame with adjusted close prices
    """
    print(f"Downloading data for {', '.join(tickers)}...")
    data = yf.download(tickers, start=start_date)['Adj Close']
    data.ffill(inplace=True)
    print(f"✓ Downloaded {len(data)} records")
    return data


def plot_crypto_prices(data, tickers):
    """Plot cryptocurrency price trends on log scale"""
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(data[ticker], label=ticker, linewidth=2)
    plt.title('Cryptocurrency Prices (Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD - Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crypto_prices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Crypto prices chart saved as 'crypto_prices.png'")


def calculate_and_plot_returns(data, tickers):
    """Calculate and visualize daily log returns"""
    returns = np.log(data / data.shift(1))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Returns over time
    for ticker in tickers:
        axes[0].plot(returns[ticker], label=ticker, alpha=0.7)
    axes[0].set_title('Daily Log Returns', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Log Return')
    axes[0].legend(loc='best')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Returns distribution
    returns.dropna().hist(bins=50, alpha=0.7, ax=axes[1], figsize=(14, 5))
    axes[1].set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Log Return')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(tickers, loc='best')
    
    plt.tight_layout()
    plt.savefig('crypto_returns.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Crypto returns chart saved as 'crypto_returns.png'")
    
    return returns


def calculate_and_plot_volatility(returns, tickers, window=30):
    """Calculate and visualize rolling volatility"""
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(volatility[ticker], label=ticker, linewidth=2)
    plt.title(f'{window}-Day Rolling Volatility (Annualized)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crypto_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Crypto volatility chart saved as 'crypto_volatility.png'")
    
    # Print volatility statistics
    print("\n" + "="*80)
    print("VOLATILITY STATISTICS (Annualized)")
    print("="*80)
    for ticker in tickers:
        vol_mean = volatility[ticker].mean()
        vol_max = volatility[ticker].max()
        vol_min = volatility[ticker].min()
        print(f"{ticker}:")
        print(f"  Mean: {vol_mean:.2%}  |  Max: {vol_max:.2%}  |  Min: {vol_min:.2%}")
    print("="*80 + "\n")
    
    return volatility


def train_arima_model(data, ticker='BTC-USD', order=(5, 1, 0), train_ratio=0.9):
    """
    Train ARIMA model for price forecasting
    
    Args:
        data: Price data
        ticker: Cryptocurrency ticker
        order: ARIMA order (p, d, q)
        train_ratio: Proportion of data for training
    
    Returns:
        train, test, predictions
    """
    print(f"\nTraining ARIMA{order} model for {ticker}...")
    
    series = data[ticker].dropna()
    train_size = int(len(series) * train_ratio)
    train, test = series[:train_size], series[train_size:]
    
    # Fit model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    # Forecast
    predictions = model_fit.forecast(steps=len(test))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    print(f"✓ Model trained")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return train, test, predictions


def plot_forecast(train, test, predictions, ticker='BTC-USD'):
    """Plot actual vs forecasted prices"""
    plt.figure(figsize=(14, 7))
    
    # Plot last 100 days of training data for context
    plt.plot(train.index[-100:], train[-100:], label='Training Data (Last 100 days)', 
            color='blue', linewidth=2)
    plt.plot(test.index, test, label='Actual Prices', color='green', linewidth=2)
    plt.plot(test.index, predictions, label='Forecasted Prices', 
            color='red', linestyle='--', linewidth=2)
    
    plt.title(f'{ticker} Price Forecast using ARIMA', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{ticker.replace("-", "_")}_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Forecast chart saved as '{ticker.replace('-', '_')}_forecast.png'")


def calculate_correlation_matrix(returns, tickers):
    """Calculate and visualize correlation between cryptocurrencies"""
    corr_matrix = returns[tickers].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Cryptocurrency Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('crypto_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Correlation matrix saved as 'crypto_correlation.png'")
    
    return corr_matrix


def calculate_value_at_risk(returns, ticker='BTC-USD', confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) using historical simulation
    
    Args:
        returns: Returns data
        ticker: Cryptocurrency ticker
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
    
    Returns:
        VaR value
    """
    crypto_returns = returns[ticker].dropna()
    
    # Calculate VaR
    var = crypto_returns.quantile(1 - confidence_level)
    
    # Calculate Conditional VaR (CVaR / Expected Shortfall)
    cvar = crypto_returns[crypto_returns <= var].mean()
    
    print("\n" + "="*80)
    print(f"VALUE AT RISK ANALYSIS - {ticker}")
    print("="*80)
    print(f"Confidence Level: {confidence_level:.0%}")
    print(f"{confidence_level:.0%} Daily VaR: {var:.4f} ({var:.2%})")
    print(f"Conditional VaR (CVaR): {cvar:.4f} ({cvar:.2%})")
    print(f"\nInterpretation:")
    print(f"  On any given day, there is a {(1-confidence_level):.0%} chance of losing")
    print(f"  {abs(var):.2%} or more of the investment value.")
    print("="*80 + "\n")
    
    # Visualize VaR
    plt.figure(figsize=(12, 7))
    sns.histplot(crypto_returns, bins=100, kde=True, color='skyblue')
    plt.axvline(x=var, color='red', linestyle='--', linewidth=2, 
                label=f'{confidence_level:.0%} VaR: {var:.2%}')
    plt.axvline(x=cvar, color='darkred', linestyle='--', linewidth=2,
                label=f'CVaR: {cvar:.2%}')
    plt.axvline(x=crypto_returns.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {crypto_returns.mean():.2%}')
    
    plt.title(f'{ticker} Daily Returns Distribution with VaR', fontsize=16, fontweight='bold')
    plt.xlabel('Daily Log Return', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{ticker.replace("-", "_")}_var.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ VaR chart saved as '{ticker.replace('-', '_')}_var.png'")
    
    return var, cvar


def calculate_performance_metrics(returns, tickers):
    """Calculate and display performance metrics"""
    print("\n" + "="*80)
    print("CRYPTOCURRENCY PERFORMANCE METRICS")
    print("="*80)
    
    metrics_data = []
    
    for ticker in tickers:
        r = returns[ticker].dropna()
        
        # Calculate metrics
        total_return = (np.exp(r.sum()) - 1) * 100
        annual_return = r.mean() * 252 * 100
        annual_vol = r.std() * np.sqrt(252) * 100
        sharpe = (annual_return / annual_vol) if annual_vol != 0 else 0
        max_drawdown = ((r.cumsum().expanding().max() - r.cumsum()).max()) * 100
        
        metrics_data.append({
            'Crypto': ticker,
            'Total Return': f'{total_return:.2f}%',
            'Annual Return': f'{annual_return:.2f}%',
            'Annual Volatility': f'{annual_vol:.2f}%',
            'Sharpe Ratio': f'{sharpe:.3f}',
            'Max Drawdown': f'{max_drawdown:.2f}%'
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))
    print("="*80 + "\n")
    
    return metrics_df


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CRYPTOCURRENCY PRICE ANALYSIS AND FORECASTING")
    print("="*80 + "\n")
    
    # Define cryptocurrency tickers
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
    
    # Step 1: Download data
    data = download_crypto_data(tickers)
    
    # Step 2: Plot prices
    plot_crypto_prices(data, tickers)
    
    # Step 3: Calculate and plot returns
    returns = calculate_and_plot_returns(data, tickers)
    
    # Step 4: Calculate and plot volatility
    volatility = calculate_and_plot_volatility(returns, tickers)
    
    # Step 5: Correlation analysis
    corr_matrix = calculate_correlation_matrix(returns, tickers)
    
    # Step 6: Performance metrics
    metrics_df = calculate_performance_metrics(returns, tickers)
    
    # Step 7: ARIMA forecasting for BTC
    train, test, predictions = train_arima_model(data, ticker='BTC-USD')
    plot_forecast(train, test, predictions, ticker='BTC-USD')
    
    # Step 8: Value at Risk analysis
    var_95, cvar_95 = calculate_value_at_risk(returns, ticker='BTC-USD', confidence_level=0.95)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    print("📊 Generated Files:")
    print("  - crypto_prices.png")
    print("  - crypto_returns.png")
    print("  - crypto_volatility.png")
    print("  - crypto_correlation.png")
    print("  - BTC_USD_forecast.png")
    print("  - BTC_USD_var.png")
    print("\n✅ Cryptocurrency analysis completed successfully!")
    
    print("\n💡 Key Insights:")
    print("  - Cryptocurrencies show high volatility compared to traditional assets")
    print("  - Strong correlation exists between major cryptocurrencies")
    print("  - VaR analysis reveals significant downside risk exposure")
    print("  - ARIMA models provide baseline forecasts but struggle with high volatility")


if __name__ == "__main__":
    main()
