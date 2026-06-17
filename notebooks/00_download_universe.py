import os
import pandas as pd
import numpy as np
import yfinance as yf

os.chdir(r'D:\Atharva\PROJECTS\Portfolio Risk Analyser')

UNIVERSE = {
    'TCS':           'TCS.NS',
    'Infosys':       'INFY.NS',
    'Wipro':         'WIPRO.NS',
    'HCL Tech':      'HCLTECH.NS',
    'Tech Mahindra': 'TECHM.NS',
    'HDFC Bank':     'HDFCBANK.NS',
    'ICICI Bank':    'ICICIBANK.NS',
    'Axis Bank':     'AXISBANK.NS',
    'Kotak Bank':    'KOTAKBANK.NS',
    'SBI':           'SBIN.NS',
    'Reliance':      'RELIANCE.NS',
    'ONGC':          'ONGC.NS',
    'Power Grid':    'POWERGRID.NS',
    'NTPC':          'NTPC.NS',
    'Adani Green':   'ADANIGREEN.NS',
    'HUL':           'HINDUNILVR.NS',
    'ITC':           'ITC.NS',
    'Nestle':        'NESTLEIND.NS',
    'Britannia':     'BRITANNIA.NS',
    'Sun Pharma':    'SUNPHARMA.NS',
    'Dr Reddy':      'DRREDDY.NS',
    'Cipla':         'CIPLA.NS',
    'Maruti':        'MARUTI.NS',
    'Bajaj Auto':    'BAJAJ-AUTO.NS',
    'Tata Steel':    'TATASTEEL.NS',
    'JSW Steel':     'JSWSTEEL.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'L&T':           'LT.NS',
    'Ultratech':     'ULTRACEMCO.NS',
}

print(f"Downloading {len(UNIVERSE)} stocks...")
tickers = list(UNIVERSE.values())
names = list(UNIVERSE.keys())

raw = yf.download(tickers, start='2021-01-01', end='2024-12-31', auto_adjust=True)
prices = raw['Close'].copy()
prices.columns = names
# Drop columns that failed to download entirely
prices = prices.dropna(axis=1, how='all')
# Then drop rows with any remaining NaN
prices = prices.dropna(axis=0)

print(f"Shape: {prices.shape}")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
print(f"Missing values: {prices.isnull().sum().sum()}")

os.makedirs('data', exist_ok=True)
prices.to_csv('data/universe_prices.csv')

returns = prices.pct_change().dropna()
returns.to_csv('data/universe_returns.csv')

# Sector mapping for EDA
sectors = {
    'TCS': 'IT', 'Infosys': 'IT', 'Wipro': 'IT',
    'HCL Tech': 'IT', 'Tech Mahindra': 'IT',
    'HDFC Bank': 'Banking', 'ICICI Bank': 'Banking',
    'Axis Bank': 'Banking', 'Kotak Bank': 'Banking', 'SBI': 'Banking',
    'Reliance': 'Energy', 'ONGC': 'Energy', 'Power Grid': 'Energy',
    'NTPC': 'Energy', 'Adani Green': 'Energy',
    'HUL': 'FMCG', 'ITC': 'FMCG', 'Nestle': 'FMCG', 'Britannia': 'FMCG',
    'Sun Pharma': 'Pharma', 'Dr Reddy': 'Pharma', 'Cipla': 'Pharma',
    'Tata Motors': 'Auto', 'Maruti': 'Auto', 'Bajaj Auto': 'Auto',
    'Tata Steel': 'Metals', 'JSW Steel': 'Metals',
    'Bharti Airtel': 'Telecom',
    'L&T': 'Infra', 'Ultratech': 'Infra',
}
pd.Series(sectors, name='Sector').to_csv('data/sectors.csv', header=True)

print("\nSaved:")
print("  data/universe_prices.csv")
print("  data/universe_returns.csv")
print("  data/sectors.csv")
print(f"\nAnnualised returns:")
ann_ret = (returns.mean() * 252 * 100).round(2)
print(ann_ret.sort_values(ascending=False))
print("\nDownload complete.")