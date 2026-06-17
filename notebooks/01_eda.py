# %% Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

os.chdir(r'D:\Atharva\PROJECTS\Portfolio Risk Analyser')

prices = pd.read_csv('data/universe_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('data/universe_returns.csv', index_col=0, parse_dates=True)
sectors = pd.read_csv('data/sectors.csv', index_col=0)
sectors.columns = ['Sector']

print(f"Universe: {prices.shape[1]} stocks, {prices.shape[0]} trading days")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Sectors: {sectors['Sector'].value_counts().to_dict()}")

STOCKS = list(prices.columns)
n = len(STOCKS)

# %% EDA 1 — Normalised price performance (base 100)
print("\n--- NORMALISED PRICE PERFORMANCE ---")
normalised = prices / prices.iloc[0] * 100
print(f"Best performer: {normalised.iloc[-1].idxmax()} "
      f"({normalised.iloc[-1].max():.1f} from base 100)")
print(f"Worst performer: {normalised.iloc[-1].idxmin()} "
      f"({normalised.iloc[-1].min():.1f} from base 100)")

fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# Top 5 performers
top5 = normalised.iloc[-1].nlargest(5).index
for stock in top5:
    axes[0].plot(normalised.index, normalised[stock], label=stock, linewidth=1.5)
axes[0].axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Base (100)')
axes[0].set_title('Top 5 Performers — Normalised Price (Base 100)', fontsize=13)
axes[0].set_ylabel('Normalised Price')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Bottom 5 performers
bot5 = normalised.iloc[-1].nsmallest(5).index
for stock in bot5:
    axes[1].plot(normalised.index, normalised[stock], label=stock, linewidth=1.5)
axes[1].axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Base (100)')
axes[1].set_title('Bottom 5 Performers — Normalised Price (Base 100)', fontsize=13)
axes[1].set_ylabel('Normalised Price')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# All stocks by sector
sector_colors = {
    'IT': '#1F4E79', 'Banking': '#2E75B6', 'Energy': '#F4A460',
    'FMCG': '#1D9E75', 'Pharma': '#EC1C24', 'Auto': '#9B59B6',
    'Metals': '#7F8C8D', 'Telecom': '#E67E22', 'Infra': '#2C3E50'
}
for stock in STOCKS:
    sector = sectors.loc[stock, 'Sector'] if stock in sectors.index else 'Other'
    color = sector_colors.get(sector, 'gray')
    axes[2].plot(normalised.index, normalised[stock],
                 color=color, alpha=0.4, linewidth=0.8)
axes[2].axhline(y=100, color='black', linestyle='--', alpha=0.5)
axes[2].set_title('All Stocks — Normalised Performance by Sector', fontsize=13)
axes[2].set_ylabel('Normalised Price')
# Add sector legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=s)
                   for s, c in sector_colors.items()]
axes[2].legend(handles=legend_elements, fontsize=9, loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/charts/01_normalised_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved — normalised performance")

# %% EDA 2 — Return distribution and normality test
print("\n--- RETURN DISTRIBUTION AND FAT TAILS ---")
eq_weights = np.array([1/n] * n)
port_returns = returns @ eq_weights

# Jarque-Bera normality test
jb_stat, jb_pval = stats.jarque_bera(port_returns)
skewness = stats.skew(port_returns)
kurtosis = stats.kurtosis(port_returns)

print(f"Portfolio return statistics:")
print(f"  Mean daily return: {port_returns.mean()*100:.4f}%")
print(f"  Std daily return: {port_returns.std()*100:.4f}%")
print(f"  Skewness: {skewness:.4f} ({'negative — left tail' if skewness < 0 else 'positive'})")
print(f"  Excess kurtosis: {kurtosis:.4f} ({'fat tails — non-normal' if kurtosis > 3 else 'normal-like'})")
print(f"  Jarque-Bera stat: {jb_stat:.2f}")
print(f"  Jarque-Bera p-value: {jb_pval:.6f}")
print(f"  Normal distribution: {'REJECTED — returns are NOT normal' if jb_pval < 0.05 else 'Cannot reject normality'}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram with normal overlay
x = np.linspace(port_returns.min(), port_returns.max(), 200)
normal_pdf = stats.norm.pdf(x, port_returns.mean(), port_returns.std())
axes[0].hist(port_returns * 100, bins=100, density=True,
             color='#1F4E79', alpha=0.7, edgecolor='white', linewidth=0.2,
             label='Actual returns')
axes[0].plot(x * 100, normal_pdf / 100, 'r-', linewidth=2, label='Normal distribution')
axes[0].set_title(f'Return Distribution vs Normal\nKurtosis: {kurtosis:.2f} | JB p-value: {jb_pval:.4f}',
                  fontsize=12)
axes[0].set_xlabel('Daily Return (%)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# QQ Plot
(osm, osr), (slope, intercept, r) = stats.probplot(port_returns, dist='norm')
axes[1].scatter(osm, osr, alpha=0.5, s=8, color='#1F4E79', label='Actual')
axes[1].plot(osm, slope * np.array(osm) + intercept, 'r-',
             linewidth=2, label='Normal line')
axes[1].set_title('QQ Plot — Deviation from Normality\n(Fat tails = points above/below the line at extremes)',
                  fontsize=12)
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_ylabel('Sample Quantiles')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/charts/02_return_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved — return distribution")

# %% EDA 3 — Rolling correlation (the diversification illusion)
print("\n--- ROLLING CORRELATION ANALYSIS ---")
WINDOW = 60  # 60-day rolling window

# Calculate rolling average pairwise correlation
rolling_corr_avg = pd.Series(index=returns.index[WINDOW:], dtype=float)
for i in range(WINDOW, len(returns)):
    window_returns = returns.iloc[i-WINDOW:i]
    corr_matrix = window_returns.corr()
    upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    rolling_corr_avg.iloc[i-WINDOW] = upper_tri.mean()

print(f"Average rolling correlation: {rolling_corr_avg.mean():.4f}")
print(f"Max rolling correlation: {rolling_corr_avg.max():.4f} on {rolling_corr_avg.idxmax().date()}")
print(f"Min rolling correlation: {rolling_corr_avg.min():.4f} on {rolling_corr_avg.idxmin().date()}")
print(f"\nKEY FINDING: Correlation spikes during market stress — "
      f"diversification disappears when you need it most")

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(rolling_corr_avg.index, rolling_corr_avg, color='#1F4E79',
        linewidth=1.2, label='60-day rolling avg correlation')
ax.fill_between(rolling_corr_avg.index, rolling_corr_avg,
                rolling_corr_avg.mean(), alpha=0.2,
                where=rolling_corr_avg > rolling_corr_avg.mean(),
                color='red', label='Above average — diversification weakens')
ax.fill_between(rolling_corr_avg.index, rolling_corr_avg,
                rolling_corr_avg.mean(), alpha=0.2,
                where=rolling_corr_avg < rolling_corr_avg.mean(),
                color='green', label='Below average — diversification strong')
ax.axhline(y=rolling_corr_avg.mean(), color='black', linestyle='--',
           alpha=0.7, label=f'Average: {rolling_corr_avg.mean():.2f}')
ax.set_title('Rolling 60-Day Average Pairwise Correlation — Diversification Illusion',
             fontsize=13)
ax.set_ylabel('Average Correlation')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/charts/03_rolling_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved — rolling correlation")

# %% EDA 4 — Sector-wise performance comparison
print("\n--- SECTOR PERFORMANCE ---")
sector_returns = {}
for sector in sectors['Sector'].unique():
    sector_stocks = sectors[sectors['Sector'] == sector].index.tolist()
    sector_stocks = [s for s in sector_stocks if s in returns.columns]
    if sector_stocks:
        avg_return = returns[sector_stocks].mean(axis=1)
        sector_returns[sector] = avg_return

sector_annual = {s: (r.mean() * 252 * 100) for s, r in sector_returns.items()}
sector_vol = {s: (r.std() * np.sqrt(252) * 100) for s, r in sector_returns.items()}
print("Sector annualised returns:")
for s, r in sorted(sector_annual.items(), key=lambda x: -x[1]):
    print(f"  {s}: {r:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sectors_sorted = sorted(sector_annual.items(), key=lambda x: -x[1])
sector_names = [s[0] for s in sectors_sorted]
sector_vals = [s[1] for s in sectors_sorted]
colors = ['#1D9E75' if v > 0 else '#EC1C24' for v in sector_vals]

axes[0].barh(sector_names, sector_vals, color=colors)
axes[0].axvline(x=0, color='black', linewidth=0.8)
axes[0].set_title('Sector Annualised Returns 2021-2024', fontsize=12)
axes[0].set_xlabel('Annual Return (%)')
axes[0].grid(True, alpha=0.3, axis='x')

# Sector risk vs return scatter
sector_v_list = [sector_vol[s] for s in sector_names]
axes[1].scatter(sector_v_list, sector_vals, s=200,
                c=list(sector_colors.values())[:len(sector_names)],
                zorder=5)
for i, name in enumerate(sector_names):
    axes[1].annotate(name, (sector_v_list[i], sector_vals[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=9)
RISK_FREE_PCT = 6.5
axes[1].axhline(y=RISK_FREE_PCT, color='red', linestyle='--',
                label=f'Risk-free rate ({RISK_FREE_PCT}%)')
axes[1].set_xlabel('Annual Volatility (%)')
axes[1].set_ylabel('Annual Return (%)')
axes[1].set_title('Sector Risk vs Return', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/charts/04_sector_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved — sector performance")

# %% EDA 5 — Rolling volatility
print("\n--- ROLLING VOLATILITY ---")
WINDOW_VOL = 30
rolling_vol = returns.rolling(WINDOW_VOL).std() * np.sqrt(252) * 100
port_rolling_vol = (returns @ eq_weights).rolling(WINDOW_VOL).std() * np.sqrt(252) * 100

print(f"Current portfolio volatility (last 30 days): {port_rolling_vol.iloc[-1]:.2f}%")
print(f"Peak portfolio volatility: {port_rolling_vol.max():.2f}% on {port_rolling_vol.idxmax().date()}")
print(f"Average portfolio volatility: {port_rolling_vol.mean():.2f}%")

fig, ax = plt.subplots(figsize=(16, 6))
for stock in STOCKS:
    ax.plot(rolling_vol.index, rolling_vol[stock],
            alpha=0.15, linewidth=0.6, color='gray')
ax.plot(port_rolling_vol.index, port_rolling_vol,
        color='#1F4E79', linewidth=2.5, label='Equal-weight portfolio')
ax.set_title('30-Day Rolling Annualised Volatility — All Stocks vs Portfolio',
             fontsize=13)
ax.set_ylabel('Annualised Volatility (%)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/charts/05_rolling_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5 saved — rolling volatility")

# %% Save EDA summary
eda_summary = {
    'total_stocks': n,
    'date_range_start': str(prices.index[0].date()),
    'date_range_end': str(prices.index[-1].date()),
    'trading_days': len(prices),
    'best_performer': normalised.iloc[-1].idxmax(),
    'best_return_pct': normalised.iloc[-1].max() - 100,
    'worst_performer': normalised.iloc[-1].idxmin(),
    'worst_return_pct': normalised.iloc[-1].min() - 100,
    'portfolio_skewness': skewness,
    'portfolio_kurtosis': kurtosis,
    'jb_pvalue': jb_pval,
    'returns_normal': jb_pval > 0.05,
    'avg_rolling_corr': rolling_corr_avg.mean(),
    'max_rolling_corr': rolling_corr_avg.max(),
    'max_corr_date': str(rolling_corr_avg.idxmax().date()),
}
pd.Series(eda_summary).to_csv('data/eda_summary.csv', header=['value'])
print("\nEDA summary saved to data/eda_summary.csv")
print("\nEDA complete.")