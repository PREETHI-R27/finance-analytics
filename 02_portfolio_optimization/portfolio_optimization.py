"""
Portfolio Optimization and Risk-Return Analysis
================================================
This script implements Modern Portfolio Theory (MPT) to construct optimal portfolios
using mean-variance optimization and the efficient frontier.

Author: PREETHI R
Project: Finance Analytics Portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def download_portfolio_data(tickers, start_date='2015-01-01'):
    """Download historical data for portfolio stocks"""
    print(f"Downloading data for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start_date)['Adj Close']
    returns = data.pct_change().dropna()
    print(f"✓ Downloaded {len(data)} records")
    return data, returns


def calculate_portfolio_metrics(returns):
    """Calculate annualized returns, covariance, and volatility"""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    volatility = np.sqrt(np.diag(cov_matrix))
    
    return mean_returns, cov_matrix, volatility


def plot_risk_return_profile(mean_returns, volatility, tickers):
    """Plot risk-return scatter for individual stocks"""
    sharpe_ratios = mean_returns / volatility
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(volatility, mean_returns, c=sharpe_ratios, 
                         s=200, cmap='viridis', edgecolors='black', linewidth=1.5)
    plt.colorbar(scatter, label='Sharpe Ratio (rf=0)')
    
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (volatility[i], mean_returns[i]), 
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.title('Risk-Return Profile of Individual Stocks', fontsize=16, fontweight='bold')
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Annualized Return', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('risk_return_profile.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Risk-return profile saved as 'risk_return_profile.png'")


def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """Calculate negative Sharpe ratio for minimization"""
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std


def optimize_portfolio(mean_returns, cov_matrix, num_assets):
    """
    Find the portfolio with maximum Sharpe ratio
    
    Returns:
        optimal_weights, optimal_return, optimal_volatility, max_sharpe_ratio
    """
    print("\nOptimizing portfolio...")
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = sco.minimize(neg_sharpe_ratio, 
                         initial_guess,
                         args=(mean_returns, cov_matrix),
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
    
    optimal_weights = result.x
    optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    max_sharpe_ratio = -result.fun
    
    print("✓ Optimization complete")
    
    return optimal_weights, optimal_return, optimal_volatility, max_sharpe_ratio


def print_portfolio_allocation(tickers, weights):
    """Display optimal portfolio weights"""
    print("\n" + "="*60)
    print("OPTIMAL PORTFOLIO ALLOCATION")
    print("="*60)
    
    allocation_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights,
        'Percentage': [f"{w*100:.2f}%" for w in weights]
    }).sort_values('Weight', ascending=False)
    
    print(allocation_df.to_string(index=False))
    print("="*60)
    
    # Filter significant allocations
    significant = allocation_df[allocation_df['Weight'] > 0.01]
    
    # Pie chart
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(significant)))
    plt.pie(significant['Weight'], labels=significant['Ticker'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    plt.title('Optimal Portfolio Allocation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('portfolio_allocation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Portfolio allocation chart saved as 'portfolio_allocation.png'")


def generate_efficient_frontier(mean_returns, cov_matrix, num_assets):
    """Generate efficient frontier curve"""
    print("\nGenerating efficient frontier...")
    
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
    efficient_volatilities = []
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        
        result = sco.minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
                             num_assets * [1. / num_assets],
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
        
        if result.success:
            efficient_volatilities.append(result.fun)
        else:
            efficient_volatilities.append(np.nan)
    
    print("✓ Efficient frontier generated")
    
    return target_returns, efficient_volatilities


def plot_efficient_frontier(target_returns, efficient_volatilities, 
                            optimal_return, optimal_volatility,
                            mean_returns, volatility, tickers):
    """Plot the efficient frontier with optimal portfolio"""
    plt.figure(figsize=(14, 9))
    
    # Efficient frontier
    plt.plot(efficient_volatilities, target_returns, 
            linestyle='--', color='blue', linewidth=2.5, label='Efficient Frontier')
    
    # Individual stocks
    scatter = plt.scatter(volatility, mean_returns, 
                         c=mean_returns/volatility, s=150, 
                         cmap='viridis', edgecolors='black', linewidth=1,
                         alpha=0.7, label='Individual Stocks')
    
    # Optimal portfolio
    plt.scatter(optimal_volatility, optimal_return, 
               marker='*', color='red', s=500, 
               edgecolors='black', linewidth=2,
               label='Optimal Portfolio (Max Sharpe)', zorder=5)
    
    # Annotations for stocks
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (volatility[i], mean_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Efficient Frontier and Optimal Portfolio', fontsize=16, fontweight='bold')
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Annualized Return', fontsize=12)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Efficient frontier chart saved as 'efficient_frontier.png'")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION AND RISK-RETURN ANALYSIS")
    print("="*80 + "\n")
    
    # Define portfolio stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
              'JPM', 'V', 'JNJ', 'WMT', 'PG']
    
    # Step 1: Download data
    data, returns = download_portfolio_data(tickers)
    
    # Step 2: Calculate portfolio metrics
    mean_returns, cov_matrix, volatility = calculate_portfolio_metrics(returns)
    
    # Step 3: Plot risk-return profile
    plot_risk_return_profile(mean_returns, volatility, tickers)
    
    # Step 4: Optimize portfolio
    optimal_weights, optimal_return, optimal_volatility, max_sharpe = optimize_portfolio(
        mean_returns, cov_matrix, len(tickers)
    )
    
    # Step 5: Display allocation
    print_portfolio_allocation(tickers, optimal_weights)
    
    # Step 6: Print performance metrics
    print(f"\n{'='*60}")
    print("OPTIMAL PORTFOLIO PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Expected Annual Return:  {optimal_return:.2%}")
    print(f"Annual Volatility (Risk): {optimal_volatility:.2%}")
    print(f"Sharpe Ratio:            {max_sharpe:.4f}")
    print(f"{'='*60}\n")
    
    # Step 7: Generate efficient frontier
    target_returns, efficient_volatilities = generate_efficient_frontier(
        mean_returns, cov_matrix, len(tickers)
    )
    
    # Step 8: Plot efficient frontier
    plot_efficient_frontier(target_returns, efficient_volatilities,
                           optimal_return, optimal_volatility,
                           mean_returns, volatility, tickers)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    print("📊 Generated Files:")
    print("  - risk_return_profile.png")
    print("  - portfolio_allocation.png")
    print("  - efficient_frontier.png")
    print("\n✅ Portfolio optimization completed successfully!")


if __name__ == "__main__":
    main()
