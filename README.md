# 📈 Portfolio Risk Analyser

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![NSE](https://img.shields.io/badge/NSE-28%20Stocks-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

portfolio risk analytics and optimization system for 28 NSE large-cap stocks using historical data from 2021–2024. The platform measures portfolio risk through VaR and CDaR, while examining diversification dynamics via rolling correlation analysis. Features an interactive optimizer that supports real-time allocation adjustments and scenario-based rebalancing.

## Live Demo
🔗 [Live Streamlit App](paste-your-url-here)

---

## Key Features

| Feature | Description |
|---|---|
| Stock Selector | Build custom portfolios from 28 NSE stocks across 9 sectors |
| VaR and CVaR | Historical simulation and parametric VaR with fat-tail evidence |
| CDaR | Conditional Drawdown at Risk — captures prolonged losses VaR misses |
| Efficient Frontier | Modern Portfolio Theory optimisation with Sharpe and Min-Vol portfolios |
| Rolling Analysis | Rolling volatility, Sharpe ratio, and correlation over user-defined windows |
| What-If Rebalancer | Shift allocation between stocks and see risk metrics update instantly |
| Monte Carlo | 10,000 path simulation for 30-day portfolio value distribution |

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| Universe Overview | All 28 stocks — normalised performance, risk-return scatter, Sharpe ranking |
| Build Your Portfolio | Stock selector with equal, optimised, or custom weights |
| Risk Metrics | VaR, CVaR, CDaR, drawdown, fat-tail test, efficient frontier, Monte Carlo |
| Rolling Analysis | Rolling volatility, Sharpe, and correlation with adjustable window |
| What-If Rebalancer | Shift allocation between any two stocks — before vs after comparison |

---

## Key Findings

**1. Returns are NOT normally distributed**
Jarque-Bera test rejects normality (p-value ≈ 0). Excess kurtosis of 3.14
confirms fat tails — parametric VaR underestimates true tail risk by ~7%.
Historical simulation VaR is more reliable for this universe.

**2. The diversification illusion**
Average pairwise correlation is 0.23 — suggesting good diversification.
But correlation peaked at 0.42 on 28 February 2022 (Russia-Ukraine invasion).
Stocks became significantly more correlated during the crisis — diversification
collapses exactly when investors need it most.

**3. CDaR reveals hidden risk**
Bharti Airtel has acceptable 1-day VaR of -2.66% but CDaR of -43.47% —
meaning in sustained drawdown periods it loses far more than daily VaR suggests.
CDaR captures prolonged losses that VaR completely misses.

**4. Sector performance divergence**
Auto sector returned 32.38% annually vs Banking at 13.36% over the same period.
The 2.4x performance gap reflects RBI rate hike pressure on bank NIMs
versus post-COVID pent-up demand recovery in consumer discretionary.

**5. Optimisation bias warning**
The Max Sharpe portfolio concentrates 60% in 7 stocks based on historical returns.
This is in-sample optimisation — it fits the past, not the future.
A production portfolio would add turnover constraints and use forward-looking return estimates.

---

## Risk Metrics Explained

| Metric | What it measures | Formula |
|---|---|---|
| VaR (Historical) | Maximum loss on a bad day (1 in 20) | 5th percentile of return distribution |
| VaR (Parametric) | Same but assumes normal distribution | μ + z(0.05) × σ |
| CVaR | Average loss on worst 5% of days | Mean of returns below VaR threshold |
| CDaR | Average loss in worst 5% of sustained drawdown periods | Mean of drawdowns below drawdown threshold |
| Sharpe Ratio | Return per unit of risk | (R - Rf) / σ × √252 |
| Max Drawdown | Worst peak-to-trough loss | min((P - max(P)) / max(P)) |

---

## Portfolio Results (Equal Weight, 28 Stocks, 2021-2024)

| Metric | Value |
|---|---|
| Annual Return | 21.36% |
| Annual Volatility | 13.04% |
| Sharpe Ratio | 1.14 |
| 1-Day VaR 95% | -1.32% |
| 1-Day CVaR 95% | -1.83% |
| CDaR 95% | -9.61% |
| Max Drawdown | -13.35% |
| 30-Day VaR on ₹1Cr | ₹49,792 |

---

## Project Structure

Portfolio-Risk-Analyser/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 00_download_universe.py     # Download 28 NSE stocks via yfinance
│   ├── 01_eda.py                   # EDA — normalised performance, fat tails, rolling correlation
│   ├── 02_risk_metrics.py          # VaR, CVaR, CDaR, Sharpe, efficient frontier
│   ├── 03_optimisation.py          # Max Sharpe and Min Vol portfolio optimisation
│   └── 04_stress_test.py           # Historical crisis and Monte Carlo stress tests
├── data/
│   ├── universe_prices.csv         # 28-stock daily prices 2021-2024
│   ├── universe_returns.csv        # Daily returns (winsorised at ±10%)
│   ├── sectors.csv                 # Sector mapping for 28 stocks
│   ├── stock_metrics.csv           # Per-stock risk metrics
│   ├── portfolio_summary.csv       # Equal-weight portfolio summary
│   └── optimal_weights.csv         # Max Sharpe and Min Vol weights
└── assets/charts/                  # EDA and analysis charts

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| yfinance | NSE stock price data download |
| Pandas / NumPy | Data manipulation and numerical computation |
| SciPy | Portfolio optimisation (SLSQP) and statistical tests |
| Plotly | Interactive visualisations |
| Streamlit | Web dashboard and deployment |

---