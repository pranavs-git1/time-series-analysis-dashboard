import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


def show_eda_page(df):
    """
    Main function to display the EDA page content.
    """
    st.header("Exploratory Data Analysis (EDA)")
    st.write(
        "This section provides a deep dive into the historical data, visualizing trends, volume, and statistical properties.")

    st.subheader("Interactive Price Chart (Candlestick & Volume)")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='blue'),
                  row=2, col=1)

    fig.update_layout(
        title='Price and Trading Volume',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend Analysis: Moving Averages (50-day vs 200-day)")
    fig_ma = px.line(df, y=['Close', 'MA50', 'MA200'],
                     title='Closing Price with 50-day and 200-day Moving Averages')
    fig_ma.update_layout(yaxis_title='Price (USD)', legend_title='Metric')
    st.plotly_chart(fig_ma, use_container_width=True)

    st.subheader("Price Distribution (Daily Returns)")
    st.write(
        "This histogram shows the frequency of different daily returns. A normal distribution is centered at 0, but crypto often has 'fat tails' (extreme events).")

    fig_hist, ax = plt.subplots()
    sns.histplot(df['Daily Return'].dropna(), kde=True, ax=ax, bins=100)
    ax.set_title('Distribution of Daily Returns')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    st.pyplot(fig_hist)

    st.subheader("Feature Correlation Heatmap")
    st.write(
        "This heatmap shows how different variables relate to each other. Values close to 1 or -1 indicate a strong positive or negative correlation.")

    corr_df = df[['Close', 'Volume', 'Daily Return', 'MA50', 'MA200', 'Volatility', 'Sentiment Score']]
    corr_matrix = corr_df.corr()

    fig_heatmap, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix of Key Features')
    st.pyplot(fig_heatmap)
