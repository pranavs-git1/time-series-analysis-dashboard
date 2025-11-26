import streamlit as st
import pandas as pd
from data_loader import load_data
from eda import show_eda_page
from modeling import show_modeling_page
from sentiment import show_sentiment_page, get_simulated_sentiment

st.set_page_config(
    page_title="Crypto Forecast Dashboard",
    page_icon="₿",
    layout="wide"
)

st.sidebar.title("Project Controls")
st.sidebar.markdown("This dashboard performs Time Series Analysis of Cryptocurrencies.")

ticker = st.sidebar.selectbox("Select Cryptocurrency Ticker", ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", "XRP-USD"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2019-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

st.sidebar.markdown("---")
page = st.sidebar.radio("Select Analysis Page",
                        ["Project Overview",
                         "Exploratory Data Analysis (EDA)",
                         "Volatility & Sentiment Analysis",
                         "Time Series Forecasting Models"])

data = load_data(ticker, start_date, end_date)

if not data.empty:
    data_with_sentiment = get_simulated_sentiment(data.copy())

    if page == "Project Overview":
        st.title(f"Time Series Analysis: {ticker}")
        st.markdown(
            """
            This project focuses on analyzing cryptocurrency price trends using time series forecasting techniques.
            It leverages data analytics, statistical modeling, and machine learning to predict future price movements.

            **Use the sidebar to navigate between different analysis pages:**
            - **Exploratory Data Analysis (EDA):** Deep dive into historical data.
            - **Volatility & Sentiment Analysis:** Analyze risk and market sentiment.
            - **Time Series Forecasting Models:** Predict future prices using ARIMA and Prophet.
            """
        )
        st.subheader("Raw Data (Last 50 Days)")
        st.dataframe(data_with_sentiment.tail(50))

    elif page == "Exploratory Data Analysis (EDA)":
        show_eda_page(data_with_sentiment)

    elif page == "Volatility & Sentiment Analysis":
        show_sentiment_page(data_with_sentiment)

    elif page == "Time Series Forecasting Models":
        show_modeling_page(data_with_sentiment)

else:
    st.warning("Please select a valid ticker and date range to begin.")
