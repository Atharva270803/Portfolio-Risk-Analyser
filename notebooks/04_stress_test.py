# %% Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'D:\Atharva\PROJECTS\Portfolio Risk Analyser')

returns = pd.read_csv('data/universe_returns.csv', index_col=0, parse_dates=True)
STOCKS = list(returns.columns)
n = len(STOCKS)
weights = np.array([1/n] * n)
port_returns = returns @ weights

print(f"Stress testing {n}-stock equal-weight portfolio")
print(f"Running stress tests...")

# %% Stress Test 1 — Historical Crisis Periods
crisis_periods = {
    'Russia-Ukraine Shock (Feb-Mar 2022)': ('2022-02-01', '2022-03-31'),
    'Rate Hike Selloff (Apr-Jun 2022)':    ('2022-04-01', '2022-06-30'),
    'Global Recession Fear (2022)':         ('2022-01-01', '2022-06-30'),
    'Adani Crisis (Jan-Feb 2023)':          ('2023-01-24', '2023-02-28'),
    'Election Volatility (Jun 2024)':       ('2024-06-01', '2024-06-10'),
}

print("\n--- HISTORICAL STRESS TESTS ---")
stress_results = []
for crisis, (start, end) in crisis_periods.items():
    try:
        cr = port_returns.loc[start:end]
        if len(cr) == 0:
            print(f"  Skipping {crisis} — no data in range")
            continue
        total_loss = (1 + cr).prod() - 1
        cum = (1 + cr).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        worst_day = cr.min()
        print(f"\n{crisis}:")
        print(f"  Total period loss: {total_loss*100:.2f}%")
        print(f"  Max drawdown in period: {max_dd*100:.2f}%")
        print(f"  Worst single day: {worst_day*100:.2f}%")
        stress_results.append({
            'Crisis': crisis,
            'Total_Loss_%': round(total_loss*100, 2),
            'Max_Drawdown_%': round(max_dd*100, 2),
            'Worst_Day_%': round(worst_day*100, 2)
        })
    except Exception as e:
        print(f"  Error on {crisis}: {e}")

# %% Stress Test 2 — Hypothetical Scenarios
print("\n--- HYPOTHETICAL SCENARIOS (1 Crore portfolio) ---")
PORTFOLIO_VALUE = 1_000_000
BETA = 0.85

scenarios = {
    'Market drops 10% in one day':   -0.10,
    'Market drops 20% (mini crash)': -0.20,
    'Market drops 30% (2008-style)': -0.30,
    'Market drops 50% (extreme)':    -0.50,
}

for scenario, shock in scenarios.items():
    port_loss = shock * BETA
    loss_inr = PORTFOLIO_VALUE * abs(port_loss)
    remaining = PORTFOLIO_VALUE * (1 + port_loss)
    print(f"\n{scenario}:")
    print(f"  Portfolio loss: {port_loss*100:.1f}%")
    print(f"  Loss on ₹1Cr: ₹{loss_inr:,.0f}")
    print(f"  Remaining: ₹{remaining:,.0f}")

# %% Stress Test 3 — Monte Carlo VaR (30 days)
print("\n--- MONTE CARLO (10,000 paths, 30 days) ---")
np.random.seed(42)
mu_daily = port_returns.mean()
sigma_daily = port_returns.std()
N_SIM = 10000
N_DAYS = 30
INIT_VAL = 1_000_000

end_vals = np.array([
    INIT_VAL * np.prod(1 + np.random.normal(mu_daily, sigma_daily, N_DAYS))
    for _ in range(N_SIM)
])

var_95_30d  = np.percentile(end_vals, 5)
cvar_95_30d = end_vals[end_vals <= var_95_30d].mean()

print(f"Initial portfolio: ₹{INIT_VAL:,.0f}")
print(f"30-day VaR  (95%): ₹{INIT_VAL - var_95_30d:,.0f} loss")
print(f"30-day CVaR (95%): ₹{INIT_VAL - cvar_95_30d:,.0f} loss")
print(f"Best case (95th):  ₹{np.percentile(end_vals, 95):,.0f}")
print(f"Expected value:    ₹{end_vals.mean():,.0f}")

# %% Monte Carlo chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(end_vals/1e5, bins=100, color='#1F4E79',
             alpha=0.7, edgecolor='white', linewidth=0.2)
axes[0].axvline(x=var_95_30d/1e5, color='orange', linestyle='--',
                linewidth=2, label=f'VaR 95%')
axes[0].axvline(x=cvar_95_30d/1e5, color='red', linestyle='-',
                linewidth=2, label=f'CVaR 95%')
axes[0].axvline(x=INIT_VAL/1e5, color='green', linestyle='--',
                linewidth=2, label='Initial value')
axes[0].set_xlabel('Portfolio Value (Lakhs ₹)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('30-Day Monte Carlo — Distribution of Outcomes')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

np.random.seed(42)
for _ in range(200):
    path = INIT_VAL * np.cumprod(
        1 + np.random.normal(mu_daily, sigma_daily, N_DAYS))
    axes[1].plot(path/1e5, alpha=0.1, linewidth=0.8, color='#1F4E79')
axes[1].axhline(y=INIT_VAL/1e5, color='green', linestyle='--',
                linewidth=2, label='Initial value')
axes[1].set_xlabel('Trading Days')
axes[1].set_ylabel('Portfolio Value (Lakhs ₹)')
axes[1].set_title('30-Day Monte Carlo — 200 Sample Paths')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/charts/07_monte_carlo.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nMonte Carlo chart saved")

# %% Save
pd.DataFrame(stress_results).to_csv('data/stress_results.csv', index=False)
print("Stress results saved")
print("\nStress testing complete.")
