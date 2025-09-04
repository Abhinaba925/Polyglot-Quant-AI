import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta

# --- UI Styling ---
def apply_css():
    """Applies the dark theme CSS for a consistent look."""
    st.markdown("""<style>
        .stApp { background-color: #0f172a; }
        .stApp > header { background: transparent; }
        div.stBlock, div.st-emotion-cache-1y4p8pa {
            background-color: #1e293b; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #334155; }
        .stButton>button {
            border-color: #2563eb;
            color: #2563eb;
        }
        .stButton>button:hover {
            border-color: #ffffff;
            color: #ffffff;
            background-color: #2563eb;
        }
        </style>""", unsafe_allow_html=True)

# --- Stock Universe ---
# A broad list of ~150 stocks from various sectors of the NSE
NSE_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFC Bank', 'ICICIBANK.NS': 'ICICI Bank',
    'INFY.NS': 'Infosys', 'HINDUNILVR.NS': 'Hindustan Unilever', 'BHARTIARTL.NS': 'Bharti Airtel', 'SBIN.NS': 'State Bank of India',
    'ITC.NS': 'ITC', 'LICI.NS': 'LIC India', 'HCLTECH.NS': 'HCL Technologies', 'LT.NS': 'Larsen & Toubro',
    'BAJFINANCE.NS': 'Bajaj Finance', 'KOTAKBANK.NS': 'Kotak Mahindra Bank', 'AXISBANK.NS': 'Axis Bank', 'MARUTI.NS': 'Maruti Suzuki',
    'ASIANPAINT.NS': 'Asian Paints', 'ADANIENT.NS': 'Adani Enterprises', 'SUNPHARMA.NS': 'Sun Pharma', 'TITAN.NS': 'Titan',
    'TATAMOTORS.NS': 'Tata Motors', 'TATASTEEL.NS': 'Tata Steel', 'ONGC.NS': 'ONGC', 'NTPC.NS': 'NTPC',
    'BAJAJFINSV.NS': 'Bajaj Finserv', 'WIPRO.NS': 'Wipro', 'POWERGRID.NS': 'Power Grid Corp', 'ULTRACEMCO.NS': 'UltraTech Cement',
    'NESTLEIND.NS': 'Nestle India', 'COALINDIA.NS': 'Coal India', 'JSWSTEEL.NS': 'JSW Steel', 'M&M.NS': 'Mahindra & Mahindra',
    'INDUSINDBK.NS': 'IndusInd Bank', 'ADANIPORTS.NS': 'Adani Ports', 'HINDALCO.NS': 'Hindalco', 'GRASIM.NS': 'Grasim',
    'TECHM.NS': 'Tech Mahindra', 'SBILIFE.NS': 'SBI Life Insurance', 'BRITANNIA.NS': 'Britannia', 'CIPLA.NS': 'Cipla',
    'DRREDDY.NS': "Dr. Reddy's Labs", 'EICHERMOT.NS': 'Eicher Motors', 'HEROMOTOCO.NS': 'Hero MotoCorp', 'APOLLOHOSP.NS': 'Apollo Hospitals',
    'SHREECEM.NS': 'Shree Cement', 'DIVISLAB.NS': "Divi's Labs", 'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'TATACONSUM.NS': 'Tata Consumer Products', 'UPL.NS': 'UPL', 'HDFCLIFE.NS': 'HDFC Life', 'BPCL.NS': 'BPCL',
    'IOC.NS': 'Indian Oil Corp', 'DLF.NS': 'DLF', 'ZOMATO.NS': 'Zomato', 'PIDILITIND.NS': 'Pidilite Industries',
    'LTIM.NS': 'LTIMindtree', 'SIEMENS.NS': 'Siemens', 'SBICARD.NS': 'SBI Cards', 'TRENT.NS': 'Trent',
    'INDIGO.NS': 'InterGlobe Aviation', 'BANKBARODA.NS': 'Bank of Baroda', 'PNB.NS': 'Punjab National Bank',
    'AMBUJACEM.NS': 'Ambuja Cements', 'ACC.NS': 'ACC', 'GAIL.NS': 'GAIL', 'VEDL.NS': 'Vedanta',
    'GODREJCP.NS': 'Godrej Consumer', 'DABUR.NS': 'Dabur', 'MARICO.NS': 'Marico', 'HAVELLS.NS': 'Havells India',
    'BERGEPAINT.NS': 'Berger Paints', 'SRF.NS': 'SRF', 'MOTHERSON.NS': 'Motherson Sumi', 'BOSCHLTD.NS': 'Bosch',
    'BEL.NS': 'Bharat Electronics', 'HAL.NS': 'Hindustan Aeronautics', 'SAIL.NS': 'SAIL', 'NMDC.NS': 'NMDC',
    'TATAPOWER.NS': 'Tata Power', 'ADANIPOWER.NS': 'Adani Power', 'IRCTC.NS': 'IRCTC', 'DMART.NS': 'Avenue Supermarts',
    'ICICIPRULI.NS': 'ICICI Pru Life', 'ICICIGI.NS': 'ICICI Lombard', 'HDFCAMC.NS': 'HDFC AMC', 'BAJAJHLDNG.NS': 'Bajaj Holdings',
    'CHOLAFIN.NS': 'Cholamandalam', 'MUTHOOTFIN.NS': 'Muthoot Finance', 'AUROPHARMA.NS': 'Aurobindo Pharma', 'LUPIN.NS': 'Lupin',
    'TORNTPHARM.NS': 'Torrent Pharma', 'BIOCON.NS': 'Biocon', 'GLAND.NS': 'Gland Pharma', 'ALKEM.NS': 'Alkem Labs',
    'MRF.NS': 'MRF', 'BALKRISIND.NS': 'Balkrishna Industries', 'JUBLFOOD.NS': 'Jubilant FoodWorks', 'NAUKRI.NS': 'Info Edge',
    'PAYTM.NS': 'One97 Communications', 'POLICYBZR.NS': 'PB Fintech', 'NYKAA.NS': 'FSN E-Commerce', 'INDUSTOWER.NS': 'Indus Towers',
    'PETRONET.NS': 'Petronet LNG', 'VOLTAS.NS': 'Voltas', 'UBL.NS': 'United Breweries', 'COLPAL.NS': 'Colgate-Palmolive',
    'PGHH.NS': 'Procter & Gamble', 'BATAINDIA.NS': 'Bata India', 'ASHOKLEY.NS': 'Ashok Leyland', 'ESCORTS.NS': 'Escorts',
    'TVSMOTOR.NS': 'TVS Motor', 'CANBK.NS': 'Canara Bank', 'IDFCFIRSTB.NS': 'IDFC First Bank', 'BANDHANBNK.NS': 'Bandhan Bank',
    'AUBANK.NS': 'AU Small Finance Bank', 'FEDERALBNK.NS': 'Federal Bank', 'MPHASIS.NS': 'Mphasis', 'PERSISTENT.NS': 'Persistent Systems',
    'COFORGE.NS': 'Coforge', 'LTTS.NS': 'L&T Technology Services', 'TATATECH.NS': 'Tata Technologies', 'DEEPAKNTR.NS': 'Deepak Nitrite',
    'AARTIIND.NS': 'Aarti Industries', 'VINATIORGA.NS': 'Vinati Organics', 'IEX.NS': 'Indian Energy Exchange', 'DIXON.NS': 'Dixon Technologies',
    'ASTRAL.NS': 'Astral', 'SUPREMEIND.NS': 'Supreme Industries', 'KAJARIACER.NS': 'Kajaria Ceramics', 'CUMMINSIND.NS': 'Cummins India',
    'ABB.NS': 'ABB India', 'POLYCAB.NS': 'Polycab India', 'KEI.NS': 'KEI Industries', 'INDIAMART.NS': 'Indiamart Intermesh',
    'JINDALSTEL.NS': 'Jindal Steel & Power', 'APLAPOLLO.NS': 'APL Apollo Tubes', 'HINDZINC.NS': 'Hindustan Zinc'
}

# --- Data Fetching and Caching ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_market_data(tickers):
    # Fetch 1 year of data to calculate 52-week highs and 200-day MAs
    data = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True)
    if data.empty:
        st.error("Could not download market data. Please try again later.")
        return None
    
    clean_data = {}
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                ticker_df = data[ticker].copy()
            else:
                ticker_df = data.copy()
            
            if not ticker_df.empty and len(ticker_df) > 200:
                # Standard Indicators
                ticker_df['MA20'] = ticker_df['Close'].rolling(window=20).mean()
                ticker_df['MA50'] = ticker_df['Close'].rolling(window=50).mean()
                ticker_df['MA200'] = ticker_df['Close'].rolling(window=200).mean()
                ticker_df['AvgVol20'] = ticker_df['Volume'].rolling(window=20).mean()
                ticker_df['RSI'] = ta.rsi(ticker_df['Close'], length=14)
                
                # MACD
                macd = ta.macd(ticker_df['Close'], fast=12, slow=26, signal=9)
                if macd is not None and not macd.empty:
                    ticker_df = ticker_df.join(macd)

                # 52-Week High/Low
                ticker_df['52W_High'] = ticker_df['Close'].rolling(window=252).max()
                ticker_df['52W_Low'] = ticker_df['Close'].rolling(window=252).min()

                clean_data[ticker] = ticker_df.iloc[-1]
        except (KeyError, IndexError):
            pass
            
    if not clean_data:
        st.error("Failed to process data for all selected tickers.")
        return None
        
    return pd.DataFrame.from_dict(clean_data, orient='index')

# --- Main Render Function ---
def render_screener():
    apply_css()
    st.title("🔎 Stock Screener")
    st.markdown("Find investment opportunities by combining multiple technical criteria to filter ~150 NSE stocks.")

    if 'screener_data' not in st.session_state:
        st.session_state.screener_data = None
    
    if st.session_state.screener_data is None:
        with st.spinner("Fetching and calculating data for all stocks... This may take a moment on first run."):
            st.session_state.screener_data = fetch_market_data(list(NSE_STOCKS.keys()))

    if st.session_state.screener_data is not None:
        data = st.session_state.screener_data

        st.subheader("Select Screening Criteria")
        
        with st.container(border=True):
            st.markdown("**Trend & Momentum Filters**")
            c1, c2, c3 = st.columns(3)
            filter_ma20 = c1.checkbox("📈 Price > 20-Day MA")
            filter_ma50 = c2.checkbox("🚀 Price > 50-Day MA")
            filter_52w_high = c3.checkbox("🏆 Near 52-Week High")

            st.markdown("**Crossover Signal Filters**")
            c1, c2 = st.columns(2)
            filter_golden_cross = c1.checkbox("🔱 Golden Cross (MA50 > MA200)")
            filter_macd_bullish = c2.checkbox(" MACD Bullish Crossover")
            
            st.markdown("**Volume & Oscillator Filters**")
            c1, c2, c3 = st.columns(3)
            filter_volume_spike = c1.checkbox("🔊 Volume Spike (>2x Avg)")
            filter_rsi_overbought = c2.checkbox("⬆️ Overbought (RSI > 70)")
            filter_rsi_oversold = c3.checkbox("⬇️ Oversold (RSI < 30)")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Apply Filters", type="primary", use_container_width=True):
                results = data.copy()
                applied_filters = []

                if filter_ma20:
                    results = results[results['Close'] > results['MA20']]
                    applied_filters.append("Price > 20d MA")
                if filter_ma50:
                    results = results[results['Close'] > results['MA50']]
                    applied_filters.append("Price > 50d MA")
                if filter_52w_high:
                    results = results[results['Close'] >= results['52W_High'] * 0.98] # Within 2% of high
                    applied_filters.append("Near 52W High")
                if filter_golden_cross:
                    results = results[results['MA50'] > results['MA200']]
                    applied_filters.append("Golden Cross")
                if filter_macd_bullish:
                    results = results[results['MACD_12_26_9'] > results['MACDs_12_26_9']]
                    applied_filters.append("MACD Bullish")
                if filter_volume_spike:
                    results = results[results['Volume'] > results['AvgVol20'] * 2.0]
                    applied_filters.append("Volume Spike")
                if filter_rsi_overbought:
                    results = results[results['RSI'] > 70]
                    applied_filters.append("RSI > 70")
                if filter_rsi_oversold:
                    results = results[results['RSI'] < 30]
                    applied_filters.append("RSI < 30")

                st.session_state.screener_results = results
                if applied_filters:
                    st.session_state.screener_title = "Screen Results: " + " & ".join(applied_filters)
                else:
                    st.session_state.screener_title = "No filters selected. Showing all stocks."
        
        if 'screener_results' in st.session_state and st.session_state.screener_results is not None:
            st.header(st.session_state.screener_title)
            results_df = st.session_state.screener_results
            
            if results_df.empty:
                st.warning("No stocks match the selected criteria.")
            else:
                results_df['Company Name'] = results_df.index.map(NSE_STOCKS)
                
                cols_to_show = ['Company Name', 'Close', 'Volume', 'RSI', 'MA50', 'MA200', '52W_High']
                display_df = results_df[[col for col in cols_to_show if col in results_df.columns]]

                formatted_df = display_df.copy()
                for col in formatted_df.columns:
                    if 'Close' in col or 'MA' in col or 'High' in col:
                        formatted_df[col] = formatted_df[col].map('₹{:,.2f}'.format)
                    elif 'Volume' in col:
                        formatted_df[col] = formatted_df[col].map('{:,.0f}'.format)
                    elif 'RSI' in col:
                        formatted_df[col] = formatted_df[col].map('{:.2f}'.format)
                
                st.dataframe(formatted_df, use_container_width=True)
                st.info(f"Found {len(results_df)} stocks matching the criteria.")
    else:
        st.error("Could not load screener data. Please refresh.")

