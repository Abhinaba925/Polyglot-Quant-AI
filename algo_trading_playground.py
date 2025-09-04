import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta

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

# --- Performance Metrics Calculation ---
def calculate_performance_metrics(returns, risk_free_rate=0.0):
    trading_days = 252
    if returns.empty or returns.isnull().all():
        return { "Annualized Return": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0, "Sortino Ratio": 0, "Max Drawdown": 0 }
    
    annualized_return = returns.mean() * trading_days
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(trading_days) if not negative_returns.empty else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    return {
        "Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio, "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown
    }

# --- Data Caching & Backtest Logic ---
@st.cache_data
def download_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"No data downloaded for {ticker}. Please check the ticker and date range.")
            return pd.DataFrame()
        
        # --- ROBUST FIX for MultiIndex and Case Inconsistency ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        
        return data
    except Exception as e:
        st.error(f"Failed to download data for {ticker}: {e}")
        return pd.DataFrame()

NIFTY50_TICKERS = {
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services', 'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys', 'ICICIBANK.NS': 'ICICI Bank', 'HINDUNILVR.NS': 'Hindustan Unilever', 'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel', 'ITC.NS': 'ITC Limited', 'LT.NS': 'Larsen & Toubro', 'BAJFINANCE.NS': 'Bajaj Finance',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank', 'HCLTECH.NS': 'HCL Technologies', 'MARUTI.NS': 'Maruti Suzuki',
    'ASIANPAINT.NS': 'Asian Paints', 'AXISBANK.NS': 'Axis Bank', 'SUNPHARMA.NS': 'Sun Pharmaceutical', 'WIPRO.NS': 'Wipro',
    'NESTLEIND.NS': 'Nestle India', 'ULTRACEMCO.NS': 'UltraTech Cement', 'TITAN.NS': 'Titan Company', 'ONGC.NS': 'ONGC',
    'NTPC.NS': 'NTPC', 'TATAMOTORS.NS': 'Tata Motors', 'POWERGRID.NS': 'Power Grid Corporation', 'BAJAJFINSV.NS': 'Bajaj Finserv',
    'INDUSINDBK.NS': 'IndusInd Bank', 'TECHM.NS': 'Tech Mahindra', 'JSWSTEEL.NS': 'JSW Steel', 'TATASTEEL.NS': 'Tata Steel',
    'ADANIPORTS.NS': 'Adani Ports', 'HINDALCO.NS': 'Hindalco Industries', 'GRASIM.NS': 'Grasim Industries',
    'DRREDDY.NS': "Dr. Reddy's Laboratories", 'CIPLA.NS': 'Cipla', 'SBILIFE.NS': 'SBI Life Insurance',
    'HDFCLIFE.NS': 'HDFC Life Insurance', 'BAJAJ-AUTO.NS': 'Bajaj Auto', 'BRITANNIA.NS': 'Britannia Industries',
    'HEROMOTOCO.NS': 'Hero MotoCorp', 'EICHERMOT.NS': 'Eicher Motors', 'COALINDIA.NS': 'Coal India',
    'DIVISLAB.NS': "Divi's Laboratories", 'UPL.NS': 'UPL', 'M&M.NS': 'Mahindra & Mahindra',
    'BPCL.NS': 'Bharat Petroleum', 'IOC.NS': 'Indian Oil Corporation', 'SHREECEM.NS': 'Shree Cement'
}

def run_strategy_backtest(data, capital, strategy_name, params, cost_params):
    df = data.copy()
    close_prices = df['close']

    # --- Signal Generation ---
    if strategy_name == 'Moving Average Crossover':
        df.ta.sma(close=close_prices, length=params['short_window'], append=True)
        df.ta.sma(close=close_prices, length=params['long_window'], append=True)
        df['signal'] = 0
        df.loc[df[f'SMA_{params["short_window"]}'] > df[f'SMA_{params["long_window"]}'], 'signal'] = 1
        df.loc[df[f'SMA_{params["short_window"]}'] < df[f'SMA_{params["long_window"]}'], 'signal'] = -1
    
    elif strategy_name == 'Mean Reversion':
        df['sma'] = close_prices.rolling(window=params['window']).mean()
        df['std'] = close_prices.rolling(window=params['window']).std()
        df['z_score'] = (close_prices - df['sma']) / df['std']
        df['signal'] = 0
        df.loc[df['z_score'] < -params['threshold'], 'signal'] = 1
        df.loc[df['z_score'] > params['threshold'], 'signal'] = -1
        df.loc[(df['z_score'].abs() < 0.5), 'signal'] = 0

    elif strategy_name == 'MACD':
        macd = df.ta.macd(close=close_prices, fast=params['fast'], slow=params['slow'], signal=params['signal_line'])
        df = pd.concat([df, macd], axis=1)
        df['signal'] = 0
        macd_line = f'MACD_{params["fast"]}_{params["slow"]}_{params["signal_line"]}'
        signal_line = f'MACDs_{params["fast"]}_{params["slow"]}_{params["signal_line"]}'
        df.loc[df[macd_line] > df[signal_line], 'signal'] = 1
        df.loc[df[macd_line] < df[signal_line], 'signal'] = -1

    elif strategy_name == 'Bollinger Bands':
        bbands = df.ta.bbands(close=close_prices, length=params['window'], std=params['std_dev'])
        df = pd.concat([df, bbands], axis=1)
        df['signal'] = 0
        lower_band = f'BBL_{params["window"]}_{params["std_dev"]}'
        upper_band = f'BBU_{params["window"]}_{params["std_dev"]}'
        df.loc[close_prices < df[lower_band], 'signal'] = 1
        df.loc[close_prices > df[upper_band], 'signal'] = -1
        middle_band = f'BBM_{params["window"]}_{params["std_dev"]}'
        df.loc[(close_prices > df[lower_band]) & (close_prices < df[upper_band]), 'signal'] = 0
        
    df['signal'] = df['signal'].fillna(0)

    # --- CORRECTED Backtest Simulation Logic ---
    cash = capital
    position_size = 0 
    last_signal = 0
    trade_log = []
    df['portfolio_value'] = capital

    for i in range(1, len(df)):
        current_price = df.at[df.index[i], 'close']
        trade_cost = cost_params['txn_cost']
        df.at[df.index[i], 'portfolio_value'] = cash + (position_size * current_price)
        current_signal = df.at[df.index[i], 'signal']

        if current_signal == 1 and last_signal != 1:
            if position_size < 0:
                cash += position_size * current_price * (1 - trade_cost)
                trade_log.append({'Date': df.index[i], 'Type': 'Cover Short', 'Price': current_price, 'Shares': -position_size, 'Cash': cash})
                position_size = 0
            
            # --- FIX: Round shares down to the nearest whole number ---
            shares_to_buy = int((cash / current_price) * (1 - trade_cost))
            if shares_to_buy > 0:
                cash -= shares_to_buy * current_price
                position_size += shares_to_buy
                trade_log.append({'Date': df.index[i], 'Type': 'Buy', 'Price': current_price, 'Shares': shares_to_buy, 'Cash': cash})

        elif current_signal == -1 and last_signal != -1:
            if position_size > 0:
                cash += position_size * current_price * (1 - trade_cost)
                trade_log.append({'Date': df.index[i], 'Type': 'Sell', 'Price': current_price, 'Shares': position_size, 'Cash': cash})
                position_size = 0
            
            # --- FIX: Round shares down to the nearest whole number ---
            shares_to_sell = int((cash / current_price) * (1 - trade_cost))
            if shares_to_sell > 0:
                cash += shares_to_sell * current_price
                position_size -= shares_to_sell
                trade_log.append({'Date': df.index[i], 'Type': 'Short', 'Price': current_price, 'Shares': shares_to_sell, 'Cash': cash})
        
        elif current_signal == 0 and last_signal != 0:
            if position_size != 0:
                trade_type = 'Sell' if position_size > 0 else 'Cover Short'
                cash += position_size * current_price * (1 - trade_cost)
                trade_log.append({'Date': df.index[i], 'Type': trade_type, 'Price': current_price, 'Shares': abs(position_size), 'Cash': cash})
                position_size = 0

        last_signal = current_signal
        df.at[df.index[i], 'portfolio_value'] = cash + (position_size * current_price)

    df['daily_return'] = df['portfolio_value'].pct_change()
    
    # --- Buy and Hold Benchmark ---
    df['b&h_shares'] = capital / df['close'].iloc[0]
    df['b&h_value'] = df['b&h_shares'] * df['close']
    df['b&h_daily_return'] = df['b&h_value'].pct_change()
    
    return df, pd.DataFrame(trade_log)


def display_results(results, trade_log):
    st.header("📈 Backtest Results")

    # --- Performance Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strategy Performance")
        strat_metrics = calculate_performance_metrics(results['daily_return'])
        st.metric("Final Portfolio Value", f"₹{results['portfolio_value'].iloc[-1]:,.2f}")
        st.metric("Annualized Return", f"{strat_metrics['Annualized Return']:.2%}")
        st.metric("Annualized Volatility", f"{strat_metrics['Annualized Volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{strat_metrics['Sharpe Ratio']:.2f}")
        st.metric("Sortino Ratio", f"{strat_metrics['Sortino Ratio']:.2f}")
        st.metric("Max Drawdown", f"{strat_metrics['Max Drawdown']:.2%}")

    with col2:
        st.subheader("Buy & Hold Performance")
        bh_metrics = calculate_performance_metrics(results['b&h_daily_return'])
        st.metric("Final B&H Value", f"₹{results['b&h_value'].iloc[-1]:,.2f}")
        st.metric("B&H Annualized Return", f"{bh_metrics['Annualized Return']:.2%}")
        st.metric("B&H Annualized Volatility", f"{bh_metrics['Annualized Volatility']:.2%}")
        st.metric("B&H Sharpe Ratio", f"{bh_metrics['Sharpe Ratio']:.2f}")
        st.metric("B&H Sortino Ratio", f"{bh_metrics['Sortino Ratio']:.2f}")
        st.metric("B&H Max Drawdown", f"{bh_metrics['Max Drawdown']:.2%}")

    # --- Equity Curve Chart ---
    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 6)); plt.style.use('seaborn-v0_8-darkgrid')
    ax.plot(results.index, results['portfolio_value'], label="Strategy", lw=2)
    ax.plot(results.index, results['b&h_value'], label="Buy & Hold", lw=2, color='orange')
    ax.set_title("Strategy Growth vs. Buy & Hold", fontsize=16, weight='bold')
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    st.pyplot(fig)

    # --- Trade Log & Signal Chart ---
    with st.expander("View Price Chart with Trade Signals"):
        fig2, ax2 = plt.subplots(figsize=(12, 6)); plt.style.use('seaborn-v0_8-darkgrid')
        ax2.plot(results.index, results['close'], label='Close Price', color='lightblue', lw=1)
        
        if not trade_log.empty:
            buy_signals = trade_log[trade_log['Type'].isin(['Buy', 'Cover Short'])]
            sell_signals = trade_log[trade_log['Type'].isin(['Sell', 'Short'])]
            ax2.scatter(buy_signals['Date'], buy_signals['Price'], marker='^', color='green', s=100, label='Buy/Cover', zorder=5)
            ax2.scatter(sell_signals['Date'], sell_signals['Price'], marker='v', color='red', s=100, label='Sell/Short', zorder=5)
        
        ax2.set_title("Price with Trading Signals", fontsize=16, weight='bold')
        ax2.legend()
        st.pyplot(fig2)

    with st.expander("View Detailed Trade Log"):
        if not trade_log.empty:
            st.dataframe(trade_log.set_index('Date'))
        else:
            st.info("No trades were executed for this strategy in the selected period.")

# --- Main Render Function ---
def render_algo_playground():
    apply_css()
    st.title("🤖 Algorithmic Trading Playground")
    st.markdown("Backtest popular trading strategies on individual stocks with realistic cost simulation.")

    with st.sidebar:
        st.header("🤖 Algo Trading Controls")

        # Stock and Date Selection
        ticker_options = [f"{ticker} ({name})" for ticker, name in NIFTY50_TICKERS.items()]
        selected_ticker_str = st.selectbox("Select a Stock", options=ticker_options, key="algo_ticker")
        selected_ticker = selected_ticker_str.split(' ')[0]

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"), min_value=pd.to_datetime("2015-01-01"), key="algo_start_date")
        with col2:
            end_date = st.date_input("End Date", pd.to_datetime("today"), key="algo_end_date")

        # Capital and Costs
        capital = st.number_input("Initial Capital (INR)", min_value=1000, value=100000, step=1000, key="algo_capital")
        transaction_cost_pct = st.number_input("Transaction Cost (%)", 0.0, value=0.1, step=0.01, key="algo_txn_cost")

        # Strategy Selection and Parameters
        st.header("📈 Strategy Setup")
        strategy = st.selectbox("Choose Strategy", ["Moving Average Crossover", "Mean Reversion", "MACD", "Bollinger Bands"], key="algo_strategy")
        
        params = {}
        if strategy == "Moving Average Crossover":
            params['short_window'] = st.slider("Short MA Window", 5, 50, 20)
            params['long_window'] = st.slider("Long MA Window", 20, 200, 50)
        elif strategy == "Mean Reversion":
            params['window'] = st.slider("Lookback Window", 10, 100, 20)
            params['threshold'] = st.slider("Z-Score Threshold", 0.5, 3.0, 1.5, 0.1)
        elif strategy == "MACD":
            params['fast'] = st.slider("Fast Period", 5, 50, 12, key="macd_fast")
            params['slow'] = st.slider("Slow Period", 20, 100, 26, key="macd_slow")
            params['signal_line'] = st.slider("Signal Line", 5, 20, 9, key="macd_signal")
        elif strategy == "Bollinger Bands":
            params['window'] = st.slider("Lookback Window", 10, 50, 20, key="bb_window")
            params['std_dev'] = st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1, key="bb_std")

        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("🚀 Run Backtest", use_container_width=True, type="primary")
        with col2:
            add_to_compare_button = st.button("Add to Comparison", use_container_width=True)

    # State Initialization
    if 'algo_saved_results' not in st.session_state:
        st.session_state.algo_saved_results = []
    if 'algo_current_result' not in st.session_state:
        st.session_state.algo_current_result = None

    # Main Logic
    if run_button:
        if start_date >= end_date:
            st.error("Error: Start Date must be before End Date.")
        else:
            data = download_data(selected_ticker, start_date, end_date)
            if not data.empty and 'close' in data.columns:
                cost_params = {'txn_cost': transaction_cost_pct / 100}
                results, trade_log = run_strategy_backtest(data, capital, strategy, params, cost_params)
                
                strat_metrics = calculate_performance_metrics(results['daily_return'])
                strat_metrics['Final Value'] = results['portfolio_value'].iloc[-1]

                st.session_state.algo_current_result = {
                    "name": f"{selected_ticker} - {strategy}",
                    "equity_curve": results['portfolio_value'],
                    "metrics": strat_metrics,
                    "trade_log": trade_log,
                    "full_results": results
                }
            elif not data.empty:
                st.error("Downloaded data is missing the 'close' column. Cannot proceed.")

    if add_to_compare_button:
        if st.session_state.algo_current_result:
            st.session_state.algo_saved_results.append(st.session_state.algo_current_result)
            st.toast(f"Added '{st.session_state.algo_current_result['name']}' to comparison.", icon="✅")
        else:
            st.warning("Please run a backtest first before adding to comparison.")

    if st.session_state.algo_current_result:
        display_results(st.session_state.algo_current_result['full_results'], st.session_state.algo_current_result['trade_log'])
    
    # --- Comparison Section ---
    if st.session_state.algo_saved_results:
        st.header("🔍 Comparison of Saved Strategies")
        fig_comp, ax_comp = plt.subplots(figsize=(12, 6)); plt.style.use('seaborn-v0_8-darkgrid')

        for res in st.session_state.algo_saved_results:
            ax_comp.plot(res['equity_curve'], label=res['name'], lw=2)
        
        ax_comp.set_title("Comparison of Strategy Equity Curves", fontsize=16, weight='bold')
        ax_comp.set_ylabel("Portfolio Value (₹)")
        ax_comp.legend()
        st.pyplot(fig_comp)
        
        summary_data = []
        for res in st.session_state.algo_saved_results:
            m = res['metrics']
            summary_data.append({
                "Strategy": res['name'],
                "Final Value": f"₹{m['Final Value']:,.0f}",
                "Return": f"{m['Annualized Return']:.2%}",
                "Volatility": f"{m['Annualized Volatility']:.2%}",
                "Sharpe Ratio": f"{m['Sharpe Ratio']:.2f}",
                "Sortino Ratio": f"{m['Sortino Ratio']:.2f}"
            })
        st.dataframe(pd.DataFrame(summary_data))
        
        if st.button("Clear Comparison Data", key="clear_algo_comp"):
            st.session_state.algo_saved_results = []
            st.session_state.algo_current_result = None
            st.rerun()

