# %% Setup
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

os.chdir(r'D:\Atharva\PROJECTS\Portfolio Risk Analyser')

returns = pd.read_csv('data/universe_returns.csv', index_col=0, parse_dates=True)
STOCKS = list(returns.columns)
n = len(STOCKS)
RISK_FREE = 0.065
print(f"Optimising portfolio of {n} stocks")
print(f"Stocks: {STOCKS}")

# %% Portfolio performance function
def portfolio_performance(weights, returns):
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = (port_return - RISK_FREE) / port_vol
    return port_return, port_vol, sharpe

# %% Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0.01, 0.40) for _ in range(n))
init_weights = np.array([1/n] * n)

# %% Optimisation 1 — Maximum Sharpe Ratio
def neg_sharpe(weights, returns):
    r, v, s = portfolio_performance(weights, returns)
    return -s

result_sharpe = minimize(neg_sharpe, init_weights,
                         args=(returns,), method='SLSQP',
                         bounds=bounds, constraints=constraints)
sharpe_weights = result_sharpe.x
sr, sv, ss = portfolio_performance(sharpe_weights, returns)

print(f"\n--- MAXIMUM SHARPE RATIO PORTFOLIO ---")
print(f"Expected Annual Return: {sr*100:.2f}%")
print(f"Expected Annual Volatility: {sv*100:.2f}%")
print(f"Sharpe Ratio: {ss:.4f}")
print(f"\nOptimal weights:")
for stock, w in zip(STOCKS, sharpe_weights):
    if w > 0.02:
        print(f"  {stock}: {w*100:.1f}%")

# %% Optimisation 2 — Minimum Volatility
def port_vol(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

result_minvol = minimize(port_vol, init_weights,
                         args=(returns,), method='SLSQP',
                         bounds=bounds, constraints=constraints)
minvol_weights = result_minvol.x
mr, mv, ms = portfolio_performance(minvol_weights, returns)

print(f"\n--- MINIMUM VOLATILITY PORTFOLIO ---")
print(f"Expected Annual Return: {mr*100:.2f}%")
print(f"Expected Annual Volatility: {mv*100:.2f}%")
print(f"Sharpe Ratio: {ms:.4f}")
print(f"\nOptimal weights:")
for stock, w in zip(STOCKS, minvol_weights):
    if w > 0.02:
        print(f"  {stock}: {w*100:.1f}%")

# %% Efficient Frontier — 5000 random portfolios
print(f"\nSimulating 5000 random portfolios...")
np.random.seed(42)
ef_returns, ef_vols, ef_sharpes = [], [], []

for _ in range(5000):
    w = np.random.random(n)
    w = w / w.sum()
    r, v, s = portfolio_performance(w, returns)
    ef_returns.append(r * 100)
    ef_vols.append(v * 100)
    ef_sharpes.append(s)

ef_returns = np.array(ef_returns)
ef_vols    = np.array(ef_vols)
ef_sharpes = np.array(ef_sharpes)

print(f"Return range: {ef_returns.min():.1f}% to {ef_returns.max():.1f}%")
print(f"Volatility range: {ef_vols.min():.1f}% to {ef_vols.max():.1f}%")
print(f"Best Sharpe in simulation: {ef_sharpes.max():.4f}")

# %% Efficient Frontier chart
eq_r, eq_v, eq_s = portfolio_performance(init_weights, returns)

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(ef_vols, ef_returns, c=ef_sharpes,
                     cmap='RdYlGn', alpha=0.5, s=8)
plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
ax.scatter(sv*100, sr*100, marker='*', color='gold',
           s=400, zorder=5, label=f'Max Sharpe ({ss:.2f})')
ax.scatter(mv*100, mr*100, marker='D', color='blue',
           s=150, zorder=5, label=f'Min Vol ({ms:.2f} Sharpe)')
ax.scatter(eq_v*100, eq_r*100, marker='o', color='red',
           s=150, zorder=5, label=f'Equal Weight ({eq_s:.2f} Sharpe)')
ax.set_xlabel('Annual Volatility (%)', fontsize=12)
ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_title(f'Efficient Frontier — {n}-Stock NSE Universe', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/charts/06_efficient_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Efficient Frontier chart saved")

# %% Save results
pd.Series({
    'max_sharpe_return': sr * 100,
    'max_sharpe_vol':    sv * 100,
    'max_sharpe_ratio':  ss,
    'min_vol_return':    mr * 100,
    'min_vol_vol':       mv * 100,
    'min_vol_sharpe':    ms,
    'equal_weight_return': eq_r * 100,
    'equal_weight_vol':    eq_v * 100,
    'equal_weight_sharpe': eq_s,
}).to_csv('data/optimisation_results.csv', header=['value'])

pd.DataFrame({
    'Stock':             STOCKS,
    'Max_Sharpe_%':      sharpe_weights * 100,
    'Min_Vol_%':         minvol_weights * 100,
    'Equal_Weight_%':    init_weights * 100,
}).set_index('Stock').to_csv('data/optimal_weights.csv')

print("\nOptimisation results saved.")
print("Portfolio optimisation complete.")