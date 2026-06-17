# %% Setup
import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

os.chdir(r'D:\Atharva\PROJECTS\Portfolio Risk Analyser')

# %% Define NSE stocks
# 10 stocks across 5 sectors — diversification matters for portfolio analysis
STOCKS = {
    'Reliance Industries':  'RELIANCE.NS',
    'TCS':                  'TCS.NS',
    'HDFC Bank':            'HDFCBANK.NS',
    'Infosys':              'INFY.NS',
    'ICICI Bank':           'ICICIBANK.NS',
    'Wipro':                'WIPRO.NS',
    'Axis Bank':            'AXISBANK.NS',
    'Sun Pharma':           'SUNPHARMA.NS',
    'Tata Steel':           'TATASTEEL.NS',
    'Bharti Airtel':        'BHARTIARTL.NS',
}

TICKERS = list(STOCKS.values())
NAMES   = list(STOCKS.keys())

# %% Download 3 years of daily price data
print("Downloading stock data...")
raw = yf.download(TICKERS, start='2022-01-01', end='2024-12-31',
                  auto_adjust=True)

print(f"Raw data shape: {raw.shape}")
print(f"Columns: {raw.columns.tolist()[:5]}...")

# %% Extract closing prices
prices = raw['Close'].copy()
prices.columns = NAMES
prices = prices.dropna()

print(f"\nPrice data shape: {prices.shape}")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
print(f"\nSample prices (latest 3 days):\n{prices.tail(3).round(2)}")

# %% Calculate daily returns
returns = prices.pct_change().dropna()
print(f"\nReturns shape: {returns.shape}")
print(f"\nAnnualised return (mean):")
annual_returns = (returns.mean() * 252 * 100).round(2)
print(annual_returns)

print(f"\nAnnualised volatility (std):")
annual_vol = (returns.std() * np.sqrt(252) * 100).round(2)
print(annual_vol)

# %% Save data
os.makedirs('data', exist_ok=True)
prices.to_csv('data/prices.csv')
returns.to_csv('data/returns.csv')
print("\nData saved to data/prices.csv and data/returns.csv")

# %% Plot price trends
fig = px.line(prices, title='NSE Stock Prices — 2022 to 2024',
              labels={'value': 'Price (INR)', 'variable': 'Stock'})
fig.update_layout(height=500)

import matplotlib.pyplot as plt

# Convert plotly-style data to matplotlib
fig_mpl, ax = plt.subplots(figsize=(14, 6))
for col in prices.columns:
    ax.plot(prices.index, prices[col], label=col, linewidth=1)
ax.set_title('NSE Stock Prices — 2022 to 2024', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (INR)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/charts/01_price_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved")

print("\nData fetch complete.")

