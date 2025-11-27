#  Time Series Analysis with Cryptocurrency

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Status](https://img.shields.io/badge/Status-Active-success)

##  Project Description

This project is a comprehensive **Data Science Dashboard** designed to analyze, visualize, and forecast cryptocurrency price trends. 

It leverages real-time data from Yahoo Finance to perform **Exploratory Data Analysis (EDA)**, calculates financial metrics like Volatility and Moving Averages, and uses Machine Learning (**Facebook Prophet**) to predict future price movements. Additionally, it integrates **NLP (Natural Language Processing)** concepts to demonstrate how sentiment analysis can correlate with market trends.

##  Key Features

* **Real-Time Data Collection:** Fetches live historical data for major cryptocurrencies (BTC, ETH, DOGE, SOL, XRP) using the `yfinance` API.
* **Interactive EDA Dashboard:**
    * Candlestick charts with volume data.
    * Moving Averages (50-day vs 200-day).
    * Daily Return distributions and Correlation Heatmaps.
* **Time Series Forecasting:**
    * Implements **Facebook Prophet** to generate 1-year future price predictions.
    * Visualizes trend components (weekly/yearly seasonality).
* **Volatility & Sentiment Analysis:**
    * Analyzes rolling volatility risk.
    * Includes a demo of **VADER Sentiment Analysis** on text data.
    * Simulates sentiment scores to visualize correlation with price action.
* **Robust Error Handling:** Handles missing data, delisted tickers, and API connection issues gracefully.

##  Tech Stack

* **Language:** Python
* **GUI Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
* **Machine Learning:** Facebook Prophet
* **NLP:** NLTK (Natural Language Toolkit)
* **Data Source:** yfinance

##  Project Structure

```text
crypto_dashboard/
├── app.py              # Main entry point for the Streamlit application
├── data_loader.py      # Module for fetching and cleaning data from Yahoo Finance
├── eda.py              # Module containing visualization logic for EDA
├── modeling.py         # Module containing the Prophet forecasting logic
├── sentiment.py        # Module for Sentiment Analysis and NLP demos
├── requirements.txt    # List of project dependencies
└── README.md           # Project documentation
