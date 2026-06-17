import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

st.set_page_config(
    page_title="Portfolio Risk Analyser",
    page_icon="📈",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────
RISK_FREE = 0.065

# ── Load universe data ────────────────────────────────────────
@st.cache_data
def load_universe():
    prices = pd.read_csv('data/universe_prices.csv',
                         index_col=0, parse_dates=True)
    returns = pd.read_csv('data/universe_returns.csv',
                          index_col=0, parse_dates=True)
    sectors = pd.read_csv('data/sectors.csv', index_col=0)
    sectors.columns = ['Sector']
    stock_metrics = pd.read_csv('data/stock_metrics.csv', index_col=0)
    summary = pd.read_csv('data/portfolio_summary.csv', index_col=0)
    summary.columns = ['value']
    return prices, returns, sectors, stock_metrics, summary

prices_all, returns_all, sectors_df, stock_metrics_df, summary_df = load_universe()
ALL_STOCKS = list(returns_all.columns)

# ── Helper functions ──────────────────────────────────────────
def portfolio_perf(weights, ret_df):
    r = np.sum(ret_df.mean() * weights) * 252
    v = np.sqrt(np.dot(weights.T, np.dot(ret_df.cov() * 252, weights)))
    s = (r - RISK_FREE) / v
    return r, v, s

def calc_var_cvar(port_ret, conf=0.95):
    alpha = 1 - conf
    var = np.percentile(port_ret, alpha * 100)
    cvar = port_ret[port_ret <= var].mean()
    return var, cvar

def calc_drawdown(ret_series):
    cum = (1 + ret_series).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return dd

def calc_cdar(ret_series, conf=0.95):
    dd = calc_drawdown(ret_series)
    threshold = np.percentile(dd, (1-conf)*100)
    cdar = dd[dd <= threshold].mean()
    return dd, cdar

def calc_sharpe(ret_series):
    excess = ret_series.mean() - RISK_FREE/252
    return (excess / ret_series.std()) * np.sqrt(252)

def optimise(ret_df, objective='sharpe'):
    n = ret_df.shape[1]
    init_w = np.array([1/n]*n)
    bounds = tuple((0.01, 0.40) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    if objective == 'sharpe':
        fn = lambda w: -portfolio_perf(w, ret_df)[2]
    else:
        fn = lambda w: portfolio_perf(w, ret_df)[1]
    res = minimize(fn, init_w, method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res.x

def get_sector_color(sector):
    colors = {
        'IT': '#1F4E79', 'Banking': '#2E75B6', 'Energy': '#F4A460',
        'FMCG': '#1D9E75', 'Pharma': '#EC1C24', 'Auto': '#9B59B6',
        'Metals': '#7F8C8D', 'Telecom': '#E67E22', 'Infra': '#2C3E50'
    }
    return colors.get(sector, '#AAAAAA')

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📈 Portfolio Risk Analyser")
st.sidebar.caption("28 NSE stocks | 2021-2024 | 4 years of data")

page = st.sidebar.radio("Navigate", [
    "📊 Universe Overview",
    "🔧 Build Your Portfolio",
    "📉 Risk Metrics",
    "🔄 Rolling Analysis",
    "⚡ What-If Rebalancer",
])

# ══════════════════════════════════════════════════════════════
# PAGE 1 - UNIVERSE OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Universe Overview":
    st.title("📊 NSE Stock Universe - Overview")
    st.caption("28 large-cap NSE stocks across 9 sectors | 2021-2024")

    # Sector filter
    all_sectors = ['All'] + sorted(sectors_df['Sector'].unique().tolist())
    selected_sector = st.selectbox("Filter by sector", all_sectors)

    if selected_sector == 'All':
        display_stocks = ALL_STOCKS
    else:
        display_stocks = sectors_df[sectors_df['Sector'] == selected_sector].index.tolist()
        display_stocks = [s for s in display_stocks if s in ALL_STOCKS]

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stocks", len(ALL_STOCKS))
    col2.metric("Sectors", sectors_df['Sector'].nunique())
    col3.metric("Data Period", "2021-2024")
    col4.metric("Trading Days", len(returns_all))

    st.divider()

    # Normalised performance chart
    st.subheader("Normalised Price Performance (Base = 100)")
    prices_norm = (prices_all[display_stocks] /
                   prices_all[display_stocks].iloc[0] * 100)

    fig = go.Figure()
    for stock in display_stocks:
        sector = sectors_df.loc[stock, 'Sector'] if stock in sectors_df.index else 'Other'
        color = get_sector_color(sector)
        final_val = prices_norm[stock].iloc[-1]
        fig.add_trace(go.Scatter(
            x=prices_norm.index, y=prices_norm[stock],
            name=f"{stock} ({final_val:.0f})",
            line=dict(color=color, width=1.5),
            opacity=0.8
        ))
    fig.add_hline(y=100, line_dash='dash', line_color='black', opacity=0.5)
    fig.update_layout(height=500, title='Normalised Price (Base 100 = Jan 2021)',
                      yaxis_title='Normalised Price',
                      legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Risk vs Return - All Stocks")
        metrics_display = stock_metrics_df.loc[
            [s for s in display_stocks if s in stock_metrics_df.index]
        ].copy()
        metrics_display['Sector'] = [
            sectors_df.loc[s, 'Sector'] if s in sectors_df.index else 'Other'
            for s in metrics_display.index
        ]
        fig2 = px.scatter(
            metrics_display,
            x='Ann_Vol_%', y='Ann_Return_%',
            text=metrics_display.index,
            color='Sector',
            size=[10]*len(metrics_display),
            title='Risk vs Return by Stock',
            labels={'Ann_Vol_%': 'Volatility (%)',
                    'Ann_Return_%': 'Annual Return (%)'},
            color_discrete_map={s: get_sector_color(s)
                                 for s in metrics_display['Sector'].unique()}
        )
        fig2.update_traces(textposition='top center', marker_size=10)
        fig2.add_hline(y=RISK_FREE*100, line_dash='dash',
                       line_color='red', annotation_text='Risk-free rate')
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Sharpe Ratio Ranking")
        sharpe_data = stock_metrics_df.loc[display_stocks, 'Sharpe'].sort_values(ascending=True)
        colors_list = ['#1D9E75' if v > 1 else '#F4A460' if v > 0.5
                       else '#EC1C24' for v in sharpe_data.values]
        fig3 = go.Figure(go.Bar(
            x=sharpe_data.values, y=sharpe_data.index,
            orientation='h', marker_color=colors_list
        ))
        fig3.add_vline(x=1.0, line_dash='dash', line_color='green',
                       annotation_text='Good (>1.0)')
        fig3.add_vline(x=0.5, line_dash='dash', line_color='orange',
                       annotation_text='Acceptable (>0.5)')
        fig3.update_layout(height=450, title='Sharpe Ratio by Stock',
                           xaxis_title='Sharpe Ratio')
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Full Stock Metrics Table")
    display_df = stock_metrics_df.loc[
        [s for s in display_stocks if s in stock_metrics_df.index]
    ].copy()
    if 'Sector' not in display_df.columns:
        display_df.insert(0, 'Sector', [
            sectors_df.loc[s, 'Sector'] if s in sectors_df.index else 'Other'
            for s in display_df.index
        ])
    st.dataframe(display_df.sort_values('Sharpe', ascending=False),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 - BUILD YOUR PORTFOLIO (Stock Selector)
# ══════════════════════════════════════════════════════════════
elif page == "🔧 Build Your Portfolio":
    st.title("🔧 Build Your Portfolio")
    st.caption("Select stocks and set weights - risk metrics update instantly")

    col_sel, col_opt = st.columns([1, 1])

    with col_sel:
        st.subheader("Step 1 - Select stocks")
        selected_stocks = st.multiselect(
            "Choose 3-10 stocks from 28 NSE large-caps",
            options=ALL_STOCKS,
            default=['Britannia', 'Bajaj Auto', 'HCL Tech',
                     'HDFC Bank', 'Reliance', 'Sun Pharma'],
            help="Select 3 to 10 stocks for your portfolio"
        )

    with col_opt:
        st.subheader("Step 2 - Choose optimisation")
        opt_mode = st.radio("Weight strategy", [
            "Equal weight",
            "Maximum Sharpe (optimised)",
            "Minimum Volatility (optimised)",
            "Custom weights"
        ])

    if len(selected_stocks) < 2:
        st.warning("Select at least 2 stocks to build a portfolio.")
        st.stop()

    ret_sel = returns_all[selected_stocks]
    n_sel = len(selected_stocks)

    if opt_mode == "Equal weight":
        weights = np.array([1/n_sel]*n_sel)
        weight_labels = {s: 1/n_sel for s in selected_stocks}

    elif opt_mode == "Maximum Sharpe (optimised)":
        with st.spinner("Optimising for maximum Sharpe..."):
            weights = optimise(ret_sel, 'sharpe')
        weight_labels = {s: w for s, w in zip(selected_stocks, weights)}

    elif opt_mode == "Minimum Volatility (optimised)":
        with st.spinner("Optimising for minimum volatility..."):
            weights = optimise(ret_sel, 'minvol')
        weight_labels = {s: w for s, w in zip(selected_stocks, weights)}

    else:  # Custom weights
        st.subheader("Step 3 - Set custom weights")
        weight_labels = {}
        cols = st.columns(min(n_sel, 4))
        for i, stock in enumerate(selected_stocks):
            with cols[i % 4]:
                weight_labels[stock] = st.slider(
                    stock, 0, 60, int(100/n_sel), 1,
                    key=f"w_{stock}"
                ) / 100
        weights = np.array([weight_labels[s] for s in selected_stocks])
        total = weights.sum()
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total*100:.1f}% - normalising to 100%")
            weights = weights / total

    # Calculate metrics
    port_ret = ret_sel @ weights
    ann_ret, ann_vol, sharpe = portfolio_perf(weights, ret_sel)
    var_h, cvar_h = calc_var_cvar(port_ret)
    dd, cdar = calc_cdar(port_ret)
    max_dd = dd.min()

    # Store in session state for other pages
    st.session_state['selected_stocks'] = selected_stocks
    st.session_state['weights'] = weights.tolist()

    st.divider()
    st.subheader("Portfolio Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Annual Return", f"{ann_ret*100:.2f}%")
    c2.metric("Annual Volatility", f"{ann_vol*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}",
              "Good" if sharpe > 1 else "Acceptable" if sharpe > 0.5 else "Poor")
    c4.metric("1-Day VaR 95%", f"{var_h*100:.2f}%")
    c5.metric("CDaR 95%", f"{cdar*100:.2f}%")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Portfolio Weights")
        weight_df = pd.DataFrame({
            'Stock': selected_stocks,
            'Weight %': [weights[i]*100 for i in range(n_sel)],
            'Sector': [sectors_df.loc[s, 'Sector'] if s in sectors_df.index
                       else 'Other' for s in selected_stocks]
        })
        fig_w = px.pie(weight_df, values='Weight %', names='Stock',
                       color='Sector', title='Portfolio Allocation',
                       color_discrete_map={s: get_sector_color(s)
                                           for s in weight_df['Sector'].unique()})
        st.plotly_chart(fig_w, use_container_width=True)

    with col_b:
        st.subheader("Sector Exposure")
        sector_exposure = weight_df.groupby('Sector')['Weight %'].sum().reset_index()
        fig_s = px.bar(sector_exposure, x='Sector', y='Weight %',
                       title='Sector Allocation',
                       color='Sector',
                       color_discrete_map={s: get_sector_color(s)
                                           for s in sector_exposure['Sector']})
        st.plotly_chart(fig_s, use_container_width=True)

    # Recommendations
    st.subheader("📋 Portfolio Assessment")
    issues, positives = [], []

    if sharpe >= 1.0:
        positives.append(f"Sharpe of {sharpe:.2f} above 1.0 - good risk-adjusted return")
    elif sharpe >= 0.5:
        issues.append(f"Sharpe of {sharpe:.2f} below 1.0 - mediocre risk-adjusted return")
    else:
        issues.append(f"Sharpe of {sharpe:.2f} is poor - reconsider stock selection")

    max_w = max(weights)
    max_stock = selected_stocks[np.argmax(weights)]
    if max_w > 0.35:
        issues.append(f"Over-concentrated: {max_stock} at {max_w*100:.1f}% - single stock risk")
    else:
        positives.append(f"No single stock exceeds 35% - concentration risk controlled")

    if n_sel < 5:
        issues.append(f"Only {n_sel} stocks - insufficient diversification")
    else:
        positives.append(f"{n_sel} stocks across multiple sectors - adequate diversification")

    corr = ret_sel.corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    if avg_corr > 0.6:
        issues.append(f"High avg correlation ({avg_corr:.2f}) - stocks move together")
    else:
        positives.append(f"Average correlation {avg_corr:.2f} - reasonable diversification")

    for pos in positives:
        st.success(f"✅ {pos}")
    for issue in issues:
        st.warning(f"⚠️ {issue}")

# ══════════════════════════════════════════════════════════════
# PAGE 3 - RISK METRICS
# ══════════════════════════════════════════════════════════════
elif page == "📉 Risk Metrics":
    st.title("📉 Risk Metrics - Deep Dive")

    if 'selected_stocks' not in st.session_state:
        st.info("Go to 'Build Your Portfolio' first to select your stocks.")
        selected_stocks = ALL_STOCKS[:8]
        weights = np.array([1/8]*8)
    else:
        selected_stocks = st.session_state['selected_stocks']
        weights = np.array(st.session_state['weights'])

    ret_sel = returns_all[selected_stocks]
    port_ret = ret_sel @ weights
    var_h, cvar_h = calc_var_cvar(port_ret)
    var_p = port_ret.mean() + stats.norm.ppf(0.05) * port_ret.std()
    dd, cdar = calc_cdar(port_ret)
    max_dd = dd.min()
    ann_ret, ann_vol, sharpe = portfolio_perf(weights, ret_sel)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("VaR Historical", f"{var_h*100:.2f}%")
    c2.metric("VaR Parametric", f"{var_p*100:.2f}%")
    c3.metric("CVaR", f"{cvar_h*100:.2f}%")
    c4.metric("CDaR 95%", f"{cdar*100:.2f}%")
    c5.metric("Max Drawdown", f"{max_dd*100:.2f}%")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "Return Distribution", "Drawdown", "Efficient Frontier", "Monte Carlo"
    ])

    with tab1:
        st.subheader("Return Distribution with Fat Tail Evidence")
        jb_stat, jb_pval = stats.jarque_bera(port_ret)
        kurt = stats.kurtosis(port_ret)
        skew = stats.skew(port_ret)

        col_a, col_b = st.columns(2)
        col_a.metric("Skewness", f"{skew:.4f}",
                     "Left tail" if skew < 0 else "Right tail")
        col_b.metric("Excess Kurtosis", f"{kurt:.4f}",
                     "Fat tails - non-normal" if kurt > 0 else "Thin tails")

        x = np.linspace(port_ret.min(), port_ret.max(), 200)
        normal_pdf = stats.norm.pdf(x, port_ret.mean(), port_ret.std())

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=port_ret*100, histnorm='probability density',
                                   nbinsx=80, name='Actual returns',
                                   marker_color='#1F4E79', opacity=0.7))
        fig.add_trace(go.Scatter(x=x*100, y=normal_pdf/100, mode='lines',
                                 name='Normal distribution',
                                 line=dict(color='red', width=2)))
        fig.add_vline(x=var_h*100, line_dash='dash', line_color='orange',
                      annotation_text=f'VaR Hist {var_h*100:.2f}%')
        fig.add_vline(x=cvar_h*100, line_dash='solid', line_color='darkred',
                      annotation_text=f'CVaR {cvar_h*100:.2f}%')
        fig.update_layout(title=f'Return Distribution - JB p-value: {jb_pval:.2e}',
                          xaxis_title='Daily Return (%)', height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"**Fat tail test:** Jarque-Bera p-value = {jb_pval:.2e} - "
                f"{'Normal distribution REJECTED - parametric VaR underestimates tail risk' if jb_pval < 0.05 else 'Cannot reject normality'}")

    with tab2:
        st.subheader("Portfolio Drawdown Analysis")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd*100,
            fill='tozeroy', fillcolor='rgba(236,28,36,0.25)',
            line=dict(color='#EC1C24', width=1.2),
            name='Drawdown'
        ))
        fig_dd.add_hline(y=cdar*100, line_dash='dash', line_color='darkred',
                         annotation_text=f'CDaR 95%: {cdar*100:.2f}%')
        fig_dd.add_hline(y=max_dd*100, line_dash='dash', line_color='black',
                         annotation_text=f'Max DD: {max_dd*100:.2f}%')
        fig_dd.update_layout(title='Portfolio Drawdown',
                             yaxis_title='Drawdown (%)', height=450)
        st.plotly_chart(fig_dd, use_container_width=True)

        st.info("**CDaR vs Max Drawdown:** Max drawdown shows the single worst peak-to-trough. "
                "CDaR shows the average loss across all bad drawdown periods - "
                "a more stable and reliable risk measure for portfolio management.")

    with tab3:
        st.subheader("Efficient Frontier")
        with st.spinner("Simulating 3,000 portfolios..."):
            np.random.seed(42)
            ef_r, ef_v, ef_s = [], [], []
            for _ in range(3000):
                w = np.random.random(len(selected_stocks))
                w = w / w.sum()
                r, v, s = portfolio_perf(w, ret_sel)
                ef_r.append(r*100)
                ef_v.append(v*100)
                ef_s.append(s)

        opt_w_s = optimise(ret_sel, 'sharpe')
        opt_w_v = optimise(ret_sel, 'minvol')
        sr, sv, ss = portfolio_perf(opt_w_s, ret_sel)
        mr, mv, ms = portfolio_perf(opt_w_v, ret_sel)
        cr, cv, cs = portfolio_perf(weights, ret_sel)

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=ef_v, y=ef_r, mode='markers',
            marker=dict(color=ef_s, colorscale='RdYlGn',
                        size=4, opacity=0.5,
                        colorbar=dict(title='Sharpe')),
            name='Random portfolios'
        ))
        fig_ef.add_trace(go.Scatter(
            x=[sv*100], y=[sr*100], mode='markers',
            marker=dict(symbol='star', size=20, color='gold'),
            name=f'Max Sharpe ({ss:.2f})'
        ))
        fig_ef.add_trace(go.Scatter(
            x=[mv*100], y=[mr*100], mode='markers',
            marker=dict(symbol='diamond', size=15, color='blue'),
            name=f'Min Vol ({ms:.2f})'
        ))
        fig_ef.add_trace(go.Scatter(
            x=[cv*100], y=[cr*100], mode='markers',
            marker=dict(symbol='circle', size=15, color='red'),
            name=f'Your Portfolio ({cs:.2f})'
        ))
        fig_ef.update_layout(
            title='Efficient Frontier - Your Selected Stocks',
            xaxis_title='Annual Volatility (%)',
            yaxis_title='Annual Return (%)',
            height=500
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    with tab4:
        st.subheader("Monte Carlo Simulation - 30 Days")
        investment = st.number_input("Investment amount (₹)",
                                     min_value=10000, max_value=10000000,
                                     value=1000000, step=10000)
        np.random.seed(42)
        mu_d = port_ret.mean()
        sig_d = port_ret.std()
        end_vals = np.array([
            investment * np.prod(1 + np.random.normal(mu_d, sig_d, 30))
            for _ in range(10000)
        ])
        var_30 = np.percentile(end_vals, 5)
        cvar_30 = end_vals[end_vals <= var_30].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("30-Day VaR (95%)",
                  f"₹{investment - var_30:,.0f} loss",
                  f"{(investment-var_30)/investment*100:.1f}% of portfolio")
        c2.metric("30-Day CVaR (95%)",
                  f"₹{investment - cvar_30:,.0f} loss")
        c3.metric("Expected Value after 30 days",
                  f"₹{end_vals.mean():,.0f}")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=end_vals/1e5, nbinsx=100,
            marker_color='#1F4E79', opacity=0.7,
            name='Simulated outcomes'
        ))
        fig_mc.add_vline(x=var_30/1e5, line_dash='dash', line_color='orange',
                         annotation_text='VaR 95%')
        fig_mc.add_vline(x=investment/1e5, line_dash='dash', line_color='green',
                         annotation_text='Initial value')
        fig_mc.update_layout(
            title='10,000 Monte Carlo Paths - 30-Day Portfolio Value',
            xaxis_title='Portfolio Value (Lakhs ₹)',
            height=400
        )
        st.plotly_chart(fig_mc, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 - ROLLING ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Rolling Analysis":
    st.title("🔄 Rolling Risk Analysis")
    st.caption("How has portfolio risk evolved over time?")

    if 'selected_stocks' not in st.session_state:
        selected_stocks = ALL_STOCKS[:8]
        weights = np.array([1/8]*8)
    else:
        selected_stocks = st.session_state['selected_stocks']
        weights = np.array(st.session_state['weights'])

    ret_sel = returns_all[selected_stocks]
    port_ret = ret_sel @ weights

    window = st.slider("Rolling window (trading days)", 20, 120, 60, 10)

    tab1, tab2, tab3 = st.tabs([
        "Rolling Volatility", "Rolling Sharpe", "Rolling Correlation"
    ])

    with tab1:
        st.subheader(f"Rolling {window}-Day Volatility")
        rolling_vol = port_ret.rolling(window).std() * np.sqrt(252) * 100
        stock_rolling_vols = ret_sel.rolling(window).std() * np.sqrt(252) * 100

        fig = go.Figure()
        for stock in selected_stocks:
            fig.add_trace(go.Scatter(
                x=stock_rolling_vols.index,
                y=stock_rolling_vols[stock],
                name=stock, opacity=0.3,
                line=dict(width=0.8),
                showlegend=True
            ))
        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol,
            name='Portfolio (equal weight)',
            line=dict(color='#1F4E79', width=3),
            showlegend=True
        ))
        fig.update_layout(
            title=f'Rolling {window}-Day Annualised Volatility',
            yaxis_title='Volatility (%)', height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        current_vol = rolling_vol.dropna().iloc[-1]
        avg_vol = rolling_vol.dropna().mean()
        peak_vol = rolling_vol.dropna().max()
        peak_date = rolling_vol.dropna().idxmax()

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Volatility", f"{current_vol:.2f}%",
                  f"{'Above' if current_vol > avg_vol else 'Below'} average")
        c2.metric("Average Volatility", f"{avg_vol:.2f}%")
        c3.metric("Peak Volatility", f"{peak_vol:.2f}%",
                  f"on {peak_date.strftime('%b %Y')}")

    with tab2:
        st.subheader(f"Rolling {window}-Day Sharpe Ratio")
        rolling_ret = port_ret.rolling(window).mean() * 252
        rolling_vol_raw = port_ret.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_ret - RISK_FREE) / rolling_vol_raw

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe,
            fill='tozeroy',
            fillcolor='rgba(31,78,121,0.2)',
            line=dict(color='#1F4E79', width=1.5),
            name='Rolling Sharpe'
        ))
        fig2.add_hline(y=1.0, line_dash='dash', line_color='green',
                       annotation_text='Good (1.0)')
        fig2.add_hline(y=0.5, line_dash='dash', line_color='orange',
                       annotation_text='Acceptable (0.5)')
        fig2.add_hline(y=0, line_dash='solid', line_color='red',
                       opacity=0.5, annotation_text='Breakeven')
        fig2.update_layout(
            title=f'Rolling {window}-Day Sharpe Ratio',
            yaxis_title='Sharpe Ratio', height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

        current_sharpe = rolling_sharpe.dropna().iloc[-1]
        avg_sharpe = rolling_sharpe.dropna().mean()
        negative_pct = (rolling_sharpe.dropna() < 0).mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Sharpe", f"{current_sharpe:.2f}")
        c2.metric("Average Sharpe", f"{avg_sharpe:.2f}")
        c3.metric("% Periods Negative Sharpe", f"{negative_pct:.1f}%")

    with tab3:
        st.subheader(f"Rolling {window}-Day Average Correlation")
        st.caption("The diversification illusion - correlation spikes during market stress")

        rolling_corr = pd.Series(index=ret_sel.index[window:], dtype=float)
        for i in range(window, len(ret_sel)):
            w_ret = ret_sel.iloc[i-window:i]
            c_mat = w_ret.corr()
            upper = c_mat.values[np.triu_indices_from(c_mat.values, k=1)]
            rolling_corr.iloc[i-window] = upper.mean()

        avg_corr = rolling_corr.mean()
        max_corr = rolling_corr.max()
        max_corr_date = rolling_corr.idxmax()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            line=dict(color='#1F4E79', width=1.5),
            name='Rolling avg correlation'
        ))
        fig3.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            fill='tozeroy',
            fillcolor='rgba(31,78,121,0.15)',
            line=dict(width=0), showlegend=False
        ))
        fig3.add_hline(y=avg_corr, line_dash='dash', line_color='black',
                       annotation_text=f'Average: {avg_corr:.2f}')
        fig3.update_layout(
            title=f'Rolling {window}-Day Average Pairwise Correlation',
            yaxis_title='Average Correlation', height=450
        )
        st.plotly_chart(fig3, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Correlation", f"{avg_corr:.3f}")
        c2.metric("Peak Correlation", f"{max_corr:.3f}",
                  f"on {max_corr_date.strftime('%b %Y')}")
        c3.metric("Current Correlation", f"{rolling_corr.iloc[-1]:.3f}")

        st.warning(f"**Diversification illusion:** Average correlation is {avg_corr:.2f} "
                   f"but peaked at {max_corr:.2f} in {max_corr_date.strftime('%b %Y')}. "
                   f"Stocks become more correlated during market stress - "
                   f"diversification collapses exactly when you need it most.")

# ══════════════════════════════════════════════════════════════
# PAGE 5 - WHAT-IF REBALANCER
# ══════════════════════════════════════════════════════════════
elif page == "⚡ What-If Rebalancer":
    st.title("⚡ What-If Rebalancer")
    st.caption("See how shifting allocation changes your risk metrics instantly")

    if 'selected_stocks' not in st.session_state:
        selected_stocks = ALL_STOCKS[:6]
        base_weights = np.array([1/6]*6)
    else:
        selected_stocks = st.session_state['selected_stocks']
        base_weights = np.array(st.session_state['weights'])

    ret_sel = returns_all[selected_stocks]
    n_sel = len(selected_stocks)

    # Current portfolio metrics
    port_ret_base = ret_sel @ base_weights
    base_ret, base_vol, base_sharpe = portfolio_perf(base_weights, ret_sel)
    base_var, base_cvar = calc_var_cvar(port_ret_base)
    _, base_cdar = calc_cdar(port_ret_base)

    st.subheader("Current Portfolio (from Build page)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Return", f"{base_ret*100:.2f}%")
    c2.metric("Volatility", f"{base_vol*100:.2f}%")
    c3.metric("Sharpe", f"{base_sharpe:.2f}")
    c4.metric("CVaR", f"{base_cvar*100:.2f}%")

    st.divider()
    st.subheader("Rebalance - Shift allocation between two stocks")

    col1, col2 = st.columns(2)
    with col1:
        reduce_stock = st.selectbox("Reduce allocation in:",
                                    selected_stocks, index=0)
    with col2:
        increase_stock = st.selectbox("Increase allocation in:",
                                      selected_stocks, index=min(1, n_sel-1))

    shift_pct = st.slider("Shift amount (%)", 1, 30, 10, 1)
    shift = shift_pct / 100

    if reduce_stock == increase_stock:
        st.warning("Select two different stocks.")
    else:
        reduce_idx = selected_stocks.index(reduce_stock)
        increase_idx = selected_stocks.index(increase_stock)

        new_weights = base_weights.copy()
        actual_shift = min(shift, new_weights[reduce_idx] - 0.01)
        new_weights[reduce_idx] -= actual_shift
        new_weights[increase_idx] += actual_shift
        new_weights = np.clip(new_weights, 0, 1)
        new_weights = new_weights / new_weights.sum()

        port_ret_new = ret_sel @ new_weights
        new_ret, new_vol, new_sharpe = portfolio_perf(new_weights, ret_sel)
        new_var, new_cvar = calc_var_cvar(port_ret_new)
        _, new_cdar = calc_cdar(port_ret_new)

        st.divider()
        st.subheader("Impact of Rebalancing")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Return", f"{new_ret*100:.2f}%",
                  f"{(new_ret-base_ret)*100:+.2f}%")
        c2.metric("Volatility", f"{new_vol*100:.2f}%",
                  f"{(new_vol-base_vol)*100:+.2f}%",
                  delta_color="inverse")
        c3.metric("Sharpe", f"{new_sharpe:.2f}",
                  f"{new_sharpe-base_sharpe:+.2f}")
        c4.metric("CVaR", f"{new_cvar*100:.2f}%",
                  f"{(new_cvar-base_cvar)*100:+.2f}%",
                  delta_color="inverse")

        st.divider()

        # Before vs After comparison chart
        compare_data = pd.DataFrame({
            'Metric': ['Annual Return %', 'Annual Volatility %',
                       'Sharpe Ratio', 'VaR 95% %', 'CVaR 95% %'],
            'Before': [base_ret*100, base_vol*100, base_sharpe,
                       abs(base_var)*100, abs(base_cvar)*100],
            'After':  [new_ret*100, new_vol*100, new_sharpe,
                       abs(new_var)*100, abs(new_cvar)*100]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Before', x=compare_data['Metric'],
                             y=compare_data['Before'],
                             marker_color='#7F8C8D'))
        fig.add_trace(go.Bar(name='After', x=compare_data['Metric'],
                             y=compare_data['After'],
                             marker_color='#1F4E79'))
        fig.update_layout(barmode='group',
                          title='Portfolio Metrics - Before vs After Rebalancing',
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Weight comparison
        weight_compare = pd.DataFrame({
            'Stock': selected_stocks,
            'Before %': base_weights * 100,
            'After %': new_weights * 100,
            'Change %': (new_weights - base_weights) * 100
        })
        st.subheader("Weight Changes")
        st.dataframe(weight_compare.style.map(
            lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '',
            subset=['Change %']
        ), use_container_width=True)

        # Recommendation
        st.subheader("Should you rebalance?")
        if new_sharpe > base_sharpe and new_vol <= base_vol:
            st.success(f"✅ Rebalancing improves Sharpe by {new_sharpe-base_sharpe:+.2f} "
                       f"and reduces volatility - recommended.")
        elif new_sharpe > base_sharpe and new_vol > base_vol:
            st.info(f"ℹ️ Rebalancing improves Sharpe by {new_sharpe-base_sharpe:+.2f} "
                    f"but increases volatility by {(new_vol-base_vol)*100:+.2f}% - "
                    f"depends on your risk tolerance.")
        elif new_sharpe < base_sharpe:
            st.warning(f"⚠️ Rebalancing reduces Sharpe by {new_sharpe-base_sharpe:+.2f} "
                       f"- not recommended unless you have a specific reason.")