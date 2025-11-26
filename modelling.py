import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
# from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def train_prophet(df):
    """
    Trains a Prophet model and returns the forecast.
    """
    prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    prophet_df = prophet_df.dropna()

    # Create and fit the model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    return model, forecast


# def train_arima(df):
#     """
#     Trains an ARIMA model using auto_arima to find best parameters.
#     """
#     data = df['Close']
#
#     # Split data (90% train, 10% test)
#     train_size = int(len(data) * 0.9)
#     train_data, test_data = data[:train_size], data[train_size:]
#
#     # Use auto_arima to find the best (p,d,q) order
#     # This is computationally expensive, so we cache it
#     @st.cache_data
#     def find_best_arima():
#         return auto_arima(train_data, seasonal=False, stepwise=True,
#                           suppress_warnings=True, trace=False,
#                           error_action='ignore')
#
#     with st.spinner("Finding optimal ARIMA parameters... This may take a minute."):
#         auto_model = find_best_arima()
#         st.write(f"Best ARIMA parameters found: {auto_model.order}")
#
#     # Fit the actual model
#     model = ARIMA(train_data, order=auto_model.order)
#     fitted_model = model.fit()
#
#     # Make predictions
#     predictions = fitted_model.forecast(steps=len(test_data))
#     predictions.index = test_data.index
#
#     # Calculate error
#     rmse = np.sqrt(mean_squared_error(test_data, predictions))
#
#     return train_data, test_data, predictions, rmse

def show_modeling_page(df):
    """
    Main function to display the modeling page content.
    """
    st.header("Time Series Forecasting Models")

    model_choice = st.selectbox("Select Forecasting Model", ["Prophet (Facebook)"])  # , "ARIMA"])

    if model_choice == "Prophet (Facebook)":
        st.subheader("Prophet Forecast")

        model, forecast = train_prophet(df)

        # --- Visualization 8: Prophet Forecast Plot ---
        st.write("### Forecast Plot (1 Year into the Future)")
        fig1 = plot_plotly(model, forecast)
        fig1.update_layout(yaxis_title='Price (USD)')
        st.plotly_chart(fig1, use_container_width=True)

        # --- Visualization 9: Prophet Components Plot ---
        st.write("### Forecast Components")
        st.write(
            "This shows the individual components of the forecast: the overall trend, weekly seasonality, and yearly seasonality.")
        fig2 = plot_components_plotly(model, forecast)
        st.plotly_chart(fig2, use_container_width=True)

    # elif model_choice == "ARIMA":
    #     st.subheader("ARIMA Forecast")
    #     st.write("ARIMA (Autoregressive Integrated Moving Average) is a classic statistical model for time series forecasting.")
    #
    #     # --- Visualization 10 & 11: ACF & PACF Plots ---
    #     st.write("### Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots")
    #     st.write("These plots help determine the parameters (p, q) for an ARIMA model by showing the correlation of the time series with lagged versions of itself.")
    #
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         fig_acf, ax_acf = plt.subplots()
    #         plot_acf(df['Close'].dropna(), lags=40, ax=ax_acf)
    #         ax_acf.set_title('Autocorrelation (ACF)')
    #         st.pyplot(fig_acf)
    #     with col2:
    #         fig_pacf, ax_pacf = plt.subplots()
    #         plot_pacf(df['Close'].dropna(), lags=40, ax=ax_pacf)
    #         ax_pacf.set_title('Partial Autocorrelation (PACF)')
    #         st.pyplot(fig_pacf)
    #
    #     # --- Run ARIMA and display results ---
    #     st.write("### ARIMA Model Results")
    #     train, test, predictions, rmse = train_arima(df)
    #
    #     st.metric(label="Test Set Root Mean Squared Error (RMSE)", value=f"${rmse:,.2f}")
    #
    #     # --- Visualization 12: ARIMA Forecast vs. Actual ---
    #     st.write("This chart compares the model's predictions (on the test set) against the actual prices.")
    #     plot_df = pd.DataFrame({
    #         'Actual Price': test,
    #         'Predicted Price': predictions
    #     })
    #     st.line_chart(plot_df)
