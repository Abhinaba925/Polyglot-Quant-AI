import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch
import base64
import os

# --- UI Styling ---
def apply_css():
    """Applies the dark theme CSS for a consistent look."""
    st.markdown("""<style>
        .stApp { background-color: #0f172a; }
        .stApp > header { background: transparent; }
        div.stBlock, div.st-emotion-cache-1y4p8pa {
            background-color: #1e293b; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #334155; }
        .stRadio > label { font-size: 1.1em; }
        </style>""", unsafe_allow_html=True)

# --- Optimization Functions ---
def mean_variance_optimization(mu, S):
    def objective(weights):
        portfolio_return = np.sum(mu * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        if portfolio_volatility == 0: return 0
        return -portfolio_return / portfolio_volatility
    num_assets = len(mu)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1./num_assets] * num_assets)
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def risk_parity_optimization(S):
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(S, weights))
        if portfolio_variance == 0: return 0
        marginal_contribution = np.dot(S, weights)
        risk_contribution = weights * marginal_contribution / np.sqrt(portfolio_variance)
        return np.sum((risk_contribution - risk_contribution.mean())**2)
    num_assets = S.shape[0]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1./num_assets] * num_assets)
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def hierarchical_risk_parity(returns):
    corr = returns.corr()
    cov = returns.cov()
    dist = np.sqrt((1 - corr) / 2)
    link = sch.linkage(dist, 'single')
    sort_ix = sch.leaves_list(link)
    sorted_tickers = corr.index[sort_ix]
    def get_rec_bisection(cov, sort_ix_local):
        w = pd.Series(1, index=sort_ix_local)
        c_items = [sort_ix_local]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c1_tickers, c2_tickers = c_items[i], c_items[i+1]
                c1_tickers_valid = [t for t in c1_tickers if t in cov.index]
                c2_tickers_valid = [t for t in c2_tickers if t in cov.index]
                if not c1_tickers_valid or not c2_tickers_valid: continue
                v1 = np.diag(cov.loc[c1_tickers_valid, c1_tickers_valid]).sum()
                v2 = np.diag(cov.loc[c2_tickers_valid, c2_tickers_valid]).sum()
                alpha = 1 - v1 / (v1 + v2) if (v1 + v2) != 0 else 0.5
                w[c1_tickers] *= alpha
                w[c2_tickers] *= (1 - alpha)
        return w
    weights = get_rec_bisection(cov, sorted_tickers)
    return weights.sort_index()

# --- Performance Metrics Calculation ---
def calculate_performance_metrics(returns, risk_free_rate=0.0):
    trading_days = 252
    annualized_return = returns.mean() * trading_days
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(trading_days)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    return {
        "Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio, "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown, "Calmar Ratio": calmar_ratio
    }

# --- Data Caching & Backtest Logic ---
@st.cache_data
def download_data(tickers, start, end):
    try: return yf.download(tickers, start=start, end=end)['Close']
    except Exception: return pd.DataFrame()

PREDEFINED_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'HCLTECH.NS', 'MARUTI.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'ONGC.NS', 'NTPC.NS', 'TATAMOTORS.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS', 'INDUSINDBK.NS', 'TECHM.NS']
BENCHMARK_TICKER = '^NSEI'

def run_backtest(capital, start_date, end_date, optimization_method, ticker_choice, custom_tickers, rebalance_freq, risk_free_rate, transaction_cost_pct, slippage_pct, brokerage_per_trade):
    tickers_to_use = [t.strip().upper() for t in custom_tickers.split(',')] if ticker_choice == "Custom Ticker List" else PREDEFINED_TICKERS
    data_start_date = start_date - pd.DateOffset(years=2)

    with st.spinner("Downloading historical data..."):
        prices_df = download_data(tickers_to_use + [BENCHMARK_TICKER], data_start_date, end_date)

    if prices_df.empty or BENCHMARK_TICKER not in prices_df.columns or prices_df[BENCHMARK_TICKER].isnull().all():
        st.error("Could not retrieve valid data. Please check tickers or try again.")
        return None

    benchmark_prices = prices_df[[BENCHMARK_TICKER]].dropna()
    available_tickers = [t for t in tickers_to_use if t in prices_df.columns and not prices_df[t].isnull().all()]
    stock_prices = prices_df[available_tickers].dropna()
    
    if len(available_tickers) < 2:
        st.error("Need at least two valid stocks to run optimization.")
        return None

    daily_returns = stock_prices.pct_change().dropna()
    
    portfolio_value = pd.Series(index=daily_returns.loc[start_date:end_date].index, dtype=float)
    if portfolio_value.empty:
        st.error("Not enough data for the selected date range. Please choose an earlier start or later end date.")
        return None
        
    portfolio_value.iloc[0] = capital
    current_weights = pd.Series(0, index=daily_returns.columns)
    
    rebalance_dates = []
    if rebalance_freq != "None (Buy and Hold)":
        freq_map = {"Monthly": "MS", "Quarterly": "QS", "Annually": "AS"}
        rebalance_dates = pd.date_range(start_date, end_date, freq=freq_map[rebalance_freq])
    
    # Ensure the very first day is a rebalance day
    rebalance_dates = pd.to_datetime([start_date]).union(rebalance_dates)

    total_costs = {"brokerage": 0.0, "txn_tax_slippage": 0.0, "num_trades": 0}

    with st.spinner("Running backtest simulation... This may take a while."):
        for i, date in enumerate(portfolio_value.index):
            if i > 0:
                daily_portfolio_return = (daily_returns.loc[date] * current_weights).sum()
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_portfolio_return)

            if date in rebalance_dates:
                training_end = date - pd.DateOffset(days=1)
                training_start = training_end - pd.DateOffset(years=1)
                training_returns = daily_returns.loc[training_start:training_end]

                if len(training_returns) < 60: continue

                if optimization_method == "Mean-Variance":
                    mu = training_returns.mean() * 252; S = training_returns.cov() * 252
                    new_weights_arr = mean_variance_optimization(mu, S)
                elif optimization_method == "Risk Parity":
                    S = training_returns.cov() * 252
                    new_weights_arr = risk_parity_optimization(S)
                elif optimization_method == "Hierarchical Risk Parity":
                    new_weights_arr = hierarchical_risk_parity(training_returns)
                else:
                    new_weights_arr = np.array([1./len(training_returns.columns)] * len(training_returns.columns))
                
                new_weights = pd.Series(new_weights_arr, index=training_returns.columns).fillna(0)

                turnover = (new_weights - current_weights).abs().sum() / 2
                trades_value = turnover * portfolio_value.loc[date]
                
                tax_slippage_cost = trades_value * (transaction_cost_pct / 100 + slippage_pct / 100)
                total_costs["txn_tax_slippage"] += tax_slippage_cost
                
                num_trades_now = len(new_weights[new_weights > 0.0001])
                brokerage_cost = num_trades_now * brokerage_per_trade if date == pd.to_datetime(start_date) else len(new_weights[new_weights != current_weights]) * brokerage_per_trade
                total_costs["brokerage"] += brokerage_cost
                total_costs["num_trades"] += num_trades_now
                
                portfolio_value.loc[date] -= (tax_slippage_cost + brokerage_cost)
                current_weights = new_weights

    portfolio_daily_returns = portfolio_value.pct_change().dropna()
    benchmark_returns_full = benchmark_prices.pct_change().dropna()

    common_index = portfolio_daily_returns.index.intersection(benchmark_returns_full.index)
    portfolio_daily_returns = portfolio_daily_returns.loc[common_index]
    benchmark_daily_returns = benchmark_returns_full.loc[common_index]
    
    metrics = calculate_performance_metrics(portfolio_daily_returns, risk_free_rate)
    metrics['Final Portfolio Value'] = portfolio_value.iloc[-1]
    
    return {
        "name": f"{optimization_method} ({rebalance_freq})", "metrics": metrics,
        "returns": portfolio_daily_returns, "benchmark_returns": benchmark_daily_returns.iloc[:,0],
        "weights": current_weights[current_weights > 0.0001], "latest_prices": stock_prices.loc[portfolio_value.index[-1]],
        "total_costs": total_costs
    }

def display_results(result):
    st.header("📊 Backtest Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Metrics")
        m = result['metrics']
        st.metric("Final Portfolio Value", f"₹{m['Final Portfolio Value']:,.2f}")
        st.metric("Annualized Return", f"{m['Annualized Return']:.2%}")
        st.metric("Annualized Volatility", f"{m['Annualized Volatility']:.2%}")
        st.metric("Max Drawdown", f"{m['Max Drawdown']:.2%}")
        st.metric("Sharpe Ratio", f"{m['Sharpe Ratio']:.2f}")
        st.metric("Sortino Ratio", f"{m['Sortino Ratio']:.2f}")
        
        if "total_costs" in result:
            with st.expander("View Total Transaction Costs (Entire Period)"):
                costs = result["total_costs"]
                total_cost_value = costs['brokerage'] + costs['txn_tax_slippage']
                st.metric("Total Costs Incurred", f"₹{total_cost_value:,.2f}")
                st.markdown(f"- **Total Brokerage:** ₹{costs['brokerage']:,.2f} (*for ~{costs['num_trades']} total trades*)")
                st.markdown(f"- **Total Tax & Slippage:** ₹{costs['txn_tax_slippage']:,.2f}")

    with col2:
        st.subheader("Final Portfolio Allocation")
        final_weights = result['weights'].sort_values(ascending=False)
        final_value = m['Final Portfolio Value']
        latest_prices = result['latest_prices']
        allocation_df = pd.DataFrame(final_weights).reset_index(); allocation_df.columns = ['Stock', 'Weight']
        allocation_df['Invested Value (INR)'] = allocation_df['Weight'] * final_value
        allocation_df['Latest Price (INR)'] = allocation_df['Stock'].map(latest_prices)
        allocation_df['Shares'] = (allocation_df['Invested Value (INR)'] / allocation_df['Latest Price (INR)']).astype(int)
        display_df = allocation_df.copy()
        display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
        display_df['Invested Value (INR)'] = display_df['Invested Value (INR)'].map('₹{:,.2f}'.format)
        st.dataframe(display_df[['Stock', 'Weight', 'Invested Value (INR)', 'Shares']], use_container_width=True, hide_index=True)
        
    st.subheader("Cumulative Performance")
    cumulative_portfolio = (1 + result['returns']).cumprod()
    cumulative_benchmark = (1 + result['benchmark_returns']).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6)); plt.style.use('seaborn-v0_8-darkgrid')
    ax.plot(cumulative_portfolio.index, cumulative_portfolio, label="Optimized Portfolio", lw=2.5)
    ax.plot(cumulative_benchmark.index, cumulative_benchmark, label="Nifty 50 Benchmark", lw=2.5, color='darkorange')
    ax.set_title("Portfolio Growth vs. Benchmark", fontsize=16, weight='bold'); ax.set_ylabel("Cumulative Growth (1 = Start Capital)"); ax.legend()
    st.pyplot(fig)

# --- Main Render Function for the Playground ---
def render_playground():
    apply_css()
    st.title("📈 Portfolio Optimization Playground")
    st.markdown("An advanced tool for backtesting portfolio strategies with rebalancing and realistic cost simulation.")

    with st.sidebar:
        st.header("📈 Playground Controls")
        
        # --- ADDED END DATE INPUT ---
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"), min_value=pd.to_datetime("2015-01-01"), key="pg_start_date")
        with col2:
            end_date = st.date_input("End Date", pd.to_datetime("today"), max_value=pd.to_datetime("today"), key="pg_end_date")

        capital = st.number_input("Capital (INR)", min_value=1000, value=100000, step=1000, key="pg_capital")
        
        st.header("📊 Portfolio Setup")
        optimization_method = st.selectbox("Optimization Method", ["Mean-Variance", "Risk Parity", "Hierarchical Risk Parity", "Equally Weighted"], key="pg_opt_method")
        ticker_choice = st.radio("Stock Universe", ("Predefined Nifty 50 Basket", "Custom Ticker List"), horizontal=True, key="pg_ticker_choice")
        custom_tickers = ""
        if ticker_choice == "Custom Ticker List":
            custom_tickers = st.text_area("NSE Tickers (comma-separated)", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS", key="pg_custom_tickers")

        st.header("🔄 Rebalancing Strategy")
        rebalance_freq = st.selectbox("Rebalancing Frequency", ("None (Buy and Hold)", "Monthly", "Quarterly", "Annually"), key="pg_rebalance_freq")
        
        st.header("📉 Cost & Risk Simulation")
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, value=4.0, step=0.1, key="pg_rf_rate") / 100
        transaction_cost_pct = st.number_input("Transaction Cost (%)", 0.0, value=0.1, step=0.01, key="pg_txn_cost")
        slippage_pct = st.number_input("Slippage (%)", 0.0, value=0.05, step=0.01, key="pg_slippage")
        brokerage_per_trade = st.number_input("Brokerage (INR per trade)", 0.0, value=20.0, step=1.0, key="pg_brokerage")

        col1, col2 = st.columns(2)
        with col1:
            calculate_button = st.button("🚀 Run Backtest", use_container_width=True, type="primary")
        with col2:
            add_to_compare_button = st.button("Add to Comparison", use_container_width=True)

    if 'saved_results' not in st.session_state:
        st.session_state.saved_results = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None

    if calculate_button:
        if start_date >= end_date:
            st.error("Error: Start Date must be before End Date.")
        else:
            result = run_backtest(capital, start_date, end_date, optimization_method, ticker_choice, custom_tickers, rebalance_freq, risk_free_rate, transaction_cost_pct, slippage_pct, brokerage_per_trade)
            if result:
                st.session_state.current_result = result

    if add_to_compare_button:
        if st.session_state.current_result:
            st.session_state.saved_results.append(st.session_state.current_result)
            st.toast(f"Added '{st.session_state.current_result['name']}' to comparison.", icon="✅")
        else:
            st.warning("Please run a backtest first before adding to comparison.")
    
    if st.session_state.current_result:
        display_results(st.session_state.current_result)

    if st.session_state.saved_results:
        st.header("🔍 Comparison of Saved Results")
        fig_comp, ax_comp = plt.subplots(figsize=(12, 6)); plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot benchmark only once
        if st.session_state.saved_results:
             benchmark_returns = st.session_state.saved_results[0]['benchmark_returns']
             ax_comp.plot((1 + benchmark_returns).cumprod(), label="Nifty 50 Benchmark", lw=2, color='grey', linestyle='--')

        for res in st.session_state.saved_results:
            ax_comp.plot((1 + res['returns']).cumprod(), label=res['name'], lw=2)
        
        ax_comp.set_title("Comparison of Portfolio Growth", fontsize=16, weight='bold'); ax_comp.set_ylabel("Cumulative Growth (1 = Start Capital)"); ax_comp.legend()
        st.pyplot(fig_comp)
        
        summary_data = [{"Strategy": res['name'], "Final Value": f"₹{res['metrics']['Final Portfolio Value']:,.0f}", "Return": f"{res['metrics']['Annualized Return']:.2%}", "Volatility": f"{res['metrics']['Annualized Volatility']:.2%}", "Max Drawdown": f"{res['metrics']['Max Drawdown']:.2%}", "Sharpe Ratio": f"{res['metrics']['Sharpe Ratio']:.2f}", "Sortino Ratio": f"{res['metrics']['Sortino Ratio']:.2f}"} for res in st.session_state.saved_results]
        st.dataframe(pd.DataFrame(summary_data))
        
        if st.button("Clear Comparison Data"):
            st.session_state.saved_results = []
            st.session_state.current_result = None
            st.rerun()

