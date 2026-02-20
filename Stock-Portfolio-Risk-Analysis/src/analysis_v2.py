import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_portfolio_analysis():
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../images', exist_ok=True)
    
    # Simulate Stock Data (5 Years)
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
    assets = ['AAPL', 'GOOGL', 'AMZN', 'TSLA', 'BTC']
    
    # Random walk simulation
    data = {}
    for asset in assets:
        vol = 0.02 if asset != 'BTC' else 0.05
        ret = 0.0005 if asset != 'BTC' else 0.001
        daily_rets = np.random.normal(ret, vol, len(dates))
        prices = 100 * np.exp(np.cumsum(daily_rets))
        data[asset] = prices
        
    df = pd.DataFrame(data, index=dates)
    df.to_csv('../data/portfolio_stocks_v2.csv')

    # 1. Cumulative Returns
    cum_rets = df.apply(lambda x: x / x.iloc[0])
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(cum_rets.index, cum_rets[asset], label=asset, linewidth=2)
    plt.title('Portfolio Growth: Cumulative Returns (5 Years)', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig('../images/v1_cumulative_returns.png')
    plt.close()

    # 2. Risk vs Return Scatter
    annual_rets = df.pct_change().mean() * 252
    annual_vol = df.pct_change().std() * np.sqrt(252)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(annual_vol, annual_rets, s=100, color='#00ffcc')
    for i, txt in enumerate(assets):
        plt.annotate(txt, (annual_vol[i], annual_rets[i]), xytext=(5,5), textcoords='offset points')
    plt.title('Risk-Reward Profile (Sharpe Ratio Context)', fontsize=15, fontweight='bold')
    plt.xlabel('Annual Volatility (Risk)')
    plt.ylabel('Annual Return')
    plt.savefig('../images/v2_risk_return_scatter.png')
    plt.close()

    # 3. Correlation Heatmap
    corr = df.pct_change().corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Asset Correlation Matrix', fontsize=15, fontweight='bold')
    plt.savefig('../images/v3_correlation_heatmap.png')
    plt.close()

    # 4. Monte Carlo Simulation (Next 1 Year)
    last_price = df['AAPL'].iloc[-1]
    trials = 500
    days = 252
    mu = df['AAPL'].pct_change().mean()
    sigma = df['AAPL'].pct_change().std()
    
    plt.figure(figsize=(12, 6))
    for _ in range(trials):
        sim_rets = np.random.normal(mu, sigma, days)
        sim_prices = last_price * np.exp(np.cumsum(sim_rets))
        plt.plot(sim_prices, color='green', alpha=0.1)
    plt.title('Monte Carlo Simulation: AAPL Price Path (252 Days)', fontsize=15, fontweight='bold')
    plt.savefig('../images/v4_monte_carlo.png')
    plt.close()

    # 5. Rolling Volatility
    rolling_vol = df['TSLA'].pct_change().rolling(window=30).std() * np.sqrt(252)
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_vol, color='magenta', linewidth=2)
    plt.title('30-Day Rolling Volatility (TSLA)', fontsize=15, fontweight='bold')
    plt.savefig('../images/v5_rolling_volatility.png')
    plt.close()

    print("Success: 5 quality finance visualizations generated.")

if __name__ == "__main__":
    run_portfolio_analysis()
