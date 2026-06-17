# %% Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

os.chdir(r'D:\Atharva\PROJECTS\Portfolio Risk Analyser')

prices = pd.read_csv('data/universe_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('data/universe_returns.csv', index_col=0, parse_dates=True)
sectors_df = pd.read_csv('data/sectors.csv', index_col=0)
sectors_df.columns = ['Sector']

STOCKS = list(returns.columns)
n = len(STOCKS)
RISK_FREE = 0.065
eq_weights = np.array([1/n] * n)
port_returns = returns @ eq_weights

print(f"Computing risk metrics for {n}-stock equal-weight portfolio")

# %% Core risk functions — reusable in app.py
def portfolio_performance(weights, returns_df):
    port_ret = np.sum(returns_df.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T,
               np.dot(returns_df.cov() * 252, weights)))
    sharpe = (port_ret - RISK_FREE) / port_vol
    return port_ret, port_vol, sharpe

def calculate_var_cvar(port_ret_series, confidence=0.95):
    alpha = 1 - confidence
    var = np.percentile(port_ret_series, alpha * 100)
    cvar = port_ret_series[port_ret_series <= var].mean()
    return var, cvar

def calculate_drawdown(ret_series):
    cumulative = (1 + ret_series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown

def calculate_cdar(ret_series, confidence=0.95):
    dd = calculate_drawdown(ret_series)
    alpha = 1 - confidence
    threshold = np.percentile(dd, alpha * 100)
    cdar = dd[dd <= threshold].mean()
    return dd, cdar

def calculate_sharpe(port_ret_series, risk_free=RISK_FREE):
    excess = port_ret_series.mean() - risk_free/252
    return (excess / port_ret_series.std()) * np.sqrt(252)

# %% Calculate all metrics
ann_ret, ann_vol, sharpe = portfolio_performance(eq_weights, returns)
var_h, cvar_h = calculate_var_cvar(port_returns)
var_p = port_returns.mean() + stats.norm.ppf(0.05) * port_returns.std()
drawdown, cdar = calculate_cdar(port_returns)
max_dd = drawdown.min()

print(f"\n--- EQUAL WEIGHT PORTFOLIO METRICS ---")
print(f"Annual Return:      {ann_ret*100:.2f}%")
print(f"Annual Volatility:  {ann_vol*100:.2f}%")
print(f"Sharpe Ratio:       {sharpe:.4f}")
print(f"VaR Historical:     {var_h*100:.4f}%")
print(f"VaR Parametric:     {var_p*100:.4f}%")
print(f"CVaR:               {cvar_h*100:.4f}%")
print(f"CDaR (95%):         {cdar*100:.4f}%")
print(f"Max Drawdown:       {max_dd*100:.2f}%")

# %% Per-stock risk metrics
print(f"\n--- PER STOCK RISK METRICS ---")
stock_metrics = []
for stock in STOCKS:
    sr = returns[stock]
    v, cv = calculate_var_cvar(sr)
    dd, cd = calculate_cdar(sr)
    sh = calculate_sharpe(sr)
    stock_metrics.append({
        'Stock': stock,
        'Sector': sectors_df.loc[stock, 'Sector'] if stock in sectors_df.index else 'Other',
        'Ann_Return_%': round(sr.mean() * 252 * 100, 2),
        'Ann_Vol_%': round(sr.std() * np.sqrt(252) * 100, 2),
        'Sharpe': round(sh, 4),
        'VaR_95_%': round(v * 100, 4),
        'CVaR_95_%': round(cv * 100, 4),
        'CDaR_95_%': round(cd * 100, 4),
        'Max_Drawdown_%': round(dd.min() * 100, 2),
    })

stock_metrics_df = pd.DataFrame(stock_metrics).set_index('Stock')
print(stock_metrics_df.sort_values('Sharpe', ascending=False).to_string())

# %% Efficient Frontier
print(f"\n--- EFFICIENT FRONTIER SIMULATION ---")
np.random.seed(42)
N_SIM = 5000
ef_data = []

for _ in range(N_SIM):
    w = np.random.random(n)
    w = w / w.sum()
    r, v, s = portfolio_performance(w, returns)
    ef_data.append({'return': r*100, 'vol': v*100, 'sharpe': s,
                    'weights': w.tolist()})

ef_df = pd.DataFrame(ef_data)
print(f"Return range: {ef_df['return'].min():.1f}% to {ef_df['return'].max():.1f}%")
print(f"Vol range: {ef_df['vol'].min():.1f}% to {ef_df['vol'].max():.1f}%")
print(f"Best Sharpe in simulation: {ef_df['sharpe'].max():.4f}")

# %% Optimised portfolios
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0.01, 0.40) for _ in range(n))
init_w = np.array([1/n]*n)

res_sharpe = minimize(lambda w: -portfolio_performance(w, returns)[2],
                      init_w, method='SLSQP',
                      bounds=bounds, constraints=constraints)
res_minvol = minimize(lambda w: portfolio_performance(w, returns)[1],
                      init_w, method='SLSQP',
                      bounds=bounds, constraints=constraints)

sr, sv, ss = portfolio_performance(res_sharpe.x, returns)
mr, mv, ms = portfolio_performance(res_minvol.x, returns)

print(f"\n--- OPTIMISED PORTFOLIOS ---")
print(f"Max Sharpe:   Return={sr*100:.2f}%  Vol={sv*100:.2f}%  Sharpe={ss:.4f}")
print(f"Min Vol:      Return={mr*100:.2f}%  Vol={mv*100:.2f}%  Sharpe={ms:.4f}")
print(f"Equal Weight: Return={ann_ret*100:.2f}%  Vol={ann_vol*100:.2f}%  Sharpe={sharpe:.4f}")

# %% Monte Carlo stress test
print(f"\n--- MONTE CARLO (10,000 paths, 30 days) ---")
np.random.seed(42)
mu_d = port_returns.mean()
sig_d = port_returns.std()
INIT_VAL = 1_000_000

end_vals = np.array([
    INIT_VAL * np.prod(1 + np.random.normal(mu_d, sig_d, 30))
    for _ in range(10000)
])
var_30 = np.percentile(end_vals, 5)
cvar_30 = end_vals[end_vals <= var_30].mean()

print(f"30-day VaR  (95%): ₹{INIT_VAL - var_30:,.0f} loss")
print(f"30-day CVaR (95%): ₹{INIT_VAL - cvar_30:,.0f} loss")
print(f"Expected value:    ₹{end_vals.mean():,.0f}")

# %% Save everything
os.makedirs('data', exist_ok=True)
stock_metrics_df.to_csv('data/stock_metrics.csv')
ef_df.drop('weights', axis=1).to_csv('data/efficient_frontier.csv', index=False)

portfolio_summary = {
    'ann_return': ann_ret * 100,
    'ann_vol': ann_vol * 100,
    'sharpe': sharpe,
    'var_hist': var_h * 100,
    'var_param': var_p * 100,
    'cvar': cvar_h * 100,
    'cdar': cdar * 100,
    'max_drawdown': max_dd * 100,
    'max_sharpe_return': sr * 100,
    'max_sharpe_vol': sv * 100,
    'max_sharpe_ratio': ss,
    'min_vol_return': mr * 100,
    'min_vol_vol': mv * 100,
    'min_vol_sharpe': ms,
    'mc_var_30d': INIT_VAL - var_30,
    'mc_cvar_30d': INIT_VAL - cvar_30,
}
pd.Series(portfolio_summary).to_csv('data/portfolio_summary.csv', header=['value'])

# Save optimal weights
weights_summary = pd.DataFrame({
    'Stock': STOCKS,
    'Equal_Weight_%': eq_weights * 100,
    'Max_Sharpe_%': res_sharpe.x * 100,
    'Min_Vol_%': res_minvol.x * 100,
}).set_index('Stock')
weights_summary.to_csv('data/optimal_weights.csv')

print("\nAll data saved.")
print("Risk metrics complete.")