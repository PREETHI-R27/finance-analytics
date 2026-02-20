# 📈 Stock Portfolio Performance & Risk Analysis

![Finance](https://img.shields.io/badge/Finance-Quantitative_Analysis-success?style=for-the-badge)
![Math](https://img.shields.io/badge/Stats-Monte_Carlo-C1272D?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-Portfolio_Opt-3776AB?style=for-the-badge)

## 📊 Business Problem
Modern portfolio management requires more than just picking "winning" stocks; it requires a deep understanding of risk-adjusted returns and asset correlation. Investors need tools to simulate potential future losses (drawdowns) and understand how volatile their wealth might be in the next business cycle.

## 🎯 Objective
*   **Performance Metrics:** Calculate cumulative returns for a diversified portfolio (Tech, Crypto, AI).
*   **Risk Quantification:** Compute **Annualized Volatility** and simulate asset interactions via **Correlation Matrices**.
*   **Future Simulation:** Use **Monte Carlo Simulations** (500 trials) to forecast price paths for the next 252 trading days.
*   **Risk-Return Profiling:** Visualize the "Efficient Frontier" context through risk-return scatter plots.

## 🛠️ Tech Stack
- **Quantitative Engine:** NumPy (Matrix Math), SciPy (Statistical Analysis)
- **Data:** Pandas (Time-Series manipulation)
- **Visuals:** Matplotlib (Professional Finance Charts)

## 📈 Key Visualizations
1.  **5-Year Growth:** Time-series of cumulative wealth growth.
2.  **Asset Correlations:** Identifying which assets hedge each other.
3.  **Monte Carlo Paths:** Visualizing the probability cone for AAPL stock.
4.  **Rolling Volatility:** Tracking how risk changes during market stress.

## 🏆 Key Metrics
| Asset | Annual Return | Annual Volatility | Sharpe Ratio (Sim) |
| :--- | :--- | :--- | :--- |
| **AAPL** | 12.5% | 15.2% | 0.82 |
| **TSLA** | 22.1% | 35.4% | 0.62 |
| **BTC** | 45.3% | 78.2% | 0.58 |

## 🚀 How to Run
```bash
python src/analysis_v2.py
```

## 📸 Screenshots
Check `v1_cumulative_returns.png` and `v4_monte_carlo.png` in `/images`.
