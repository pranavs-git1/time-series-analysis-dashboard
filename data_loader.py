import streamlit as st
import yfinance as yf
import pandas as pd
import traceback  


@st.cache_data(ttl=3600)  
def load_data(ticker, start_date, end_date):
    """
    Downloads historical cryptocurrency data from Yahoo Finance.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.error(f"No data found for ticker {ticker} in the selected date range.")
            return pd.DataFrame()

        if 'Close' not in data.columns:
            st.error(f"Data for {ticker} does not contain a 'Close' column. Ticker may be invalid or delisted.")
            return pd.DataFrame()


        if 'Close' in data.columns:
            data['Daily Return'] = data['Close'].pct_change()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()

        if 'Daily Return' in data.columns:
            data['Volatility'] = data['Daily Return'].rolling(window=30).std()

        if data.empty:
            st.warning("Downloaded data is empty. Please check ticker and date range.")
            return pd.DataFrame()

        return data

    except Exception as e:
        # This will catch any other unexpected errors
        st.error(f"An unexpected error occurred while loading data: {type(e).__name__} - {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
