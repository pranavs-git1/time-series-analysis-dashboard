import st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import time

st.set_page_config(
    page_title="Financial Fraud Detection Model & Dashboard",
    page_icon="🛡️",
    layout="wide"
)



@st.cache_data
def load_data(filepath):
    """Loads and performs initial timestamp conversion."""
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error(f"Error: {filepath} not found. Please run `generate_fake_data.py` first.")
        return None


@st.cache_data
def advanced_feature_engineering(df):
    """Creates advanced time-series features for each transaction."""
    st.write("Cache miss: Running advanced feature engineering...")
    df_feat = df.copy()
    df_feat = df_feat.sort_values(by=['customer_id', 'timestamp'])

    df_feat['time_since_last_tx_seconds'] = df_feat.groupby('customer_id')['timestamp'].diff().dt.total_seconds()

    df_feat_indexed = df_feat.set_index('timestamp')
    tx_count_1h = df_feat_indexed.groupby('customer_id')['transaction_id'].rolling('1H').count().shift(1)
    df_feat['tx_count_last_1h'] = tx_count_1h.reset_index(level=0, drop=True).values

    df_feat['hour_of_day'] = df_feat['timestamp'].dt.hour
    df_feat['day_of_week'] = df_feat['timestamp'].dt.dayofweek

    return df_feat.drop(['transaction_id', 'customer_id'], axis=1)


# @st.cache_resource tells Streamlit to cache the trained model
@st.cache_resource
def train_model(df):
    """Preprocesses data, trains, and evaluates a model."""
    st.write("Cache miss: Training fraud detection model...")

    X = df.drop(['is_fraud', 'timestamp'], axis=1, errors='ignore')
    y = df['is_fraud']

    categorical_features = ['location', 'device', 'day_of_week']
    numerical_features = ['amount', 'hour_of_day', 'time_since_last_tx_seconds', 'tx_count_last_1h']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
    auc_score = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    st.write("Model training complete.")
    return pipeline, report, auc_score, cm, X.columns


# Main App
st.title("Financial Fraud Detection Model & Analytics Dashboard")

# Load data
raw_data = load_data('transactions.csv')

if raw_data is not None:
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Overview", "Exploratory Data Analysis", "Fraud Model & Simulation"])

    #Page 1: Overview
    if page == "Overview":
        st.header("Dashboard Overview")

        # Calculate KPIs
        total_transactions = len(raw_data)
        total_fraud = raw_data['is_fraud'].sum()
        total_fraud_value = raw_data[raw_data['is_fraud'] == 1]['amount'].sum()
        fraud_rate = (total_fraud / total_transactions) * 100

        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_transactions:,}")
        col2.metric("Total Fraud", f"{total_fraud:,}", f"{fraud_rate:.2f}% of all")
        col3.metric("Total Fraud Value", f"${total_fraud_value:,.2f}")
        col4.metric("Avg. Fraud Value", f"${(total_fraud_value / total_fraud):,.2f}")

        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            # Viz 1: Fraud vs. Legitimate Over Time (Plotly)
            st.write("##### Fraud vs. Legitimate Transactions Over Time")
            daily_summary = raw_data.set_index('timestamp').resample('D').agg(
                fraud_count=('is_fraud', 'sum'),
                legit_count=('is_fraud', lambda x: (x == 0).sum())
            ).reset_index()
            daily_summary_melted = daily_summary.melt('timestamp', var_name='Transaction Type', value_name='Count')

            fig1 = px.line(daily_summary_melted, x='timestamp', y='Count', color='Transaction Type',
                           title="Daily Transaction Counts",
                           color_discrete_map={'fraud_count': 'red', 'legit_count': 'blue'})
            fig1.update_layout(template='plotly_dark')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Viz 2: Fraud Count by Location (Plotly)
            st.write("##### Fraud Count by Location")
            fraud_by_loc = raw_data[raw_data['is_fraud'] == 1]['location'].value_counts().reset_index()
            fig2 = px.bar(fraud_by_loc.head(10), x='location', y='count', title="Top 10 Fraud Locations",
                          color='location')
            fig2.update_layout(template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)

        # Viz 3: Fraud Amount Distribution (Plotly)
        st.write("##### Fraud vs. Legitimate Amount Distribution")
        fig3 = px.histogram(raw_data, x='amount', color='is_fraud', marginal='box',
                            title="Distribution of Transaction Amounts",
                            log_y=True, nbins=100)
        fig3.update_layout(template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)


    #EDA
    elif page == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis (EDA)")

        # Load feature-engineered data for this page
        df_feat = advanced_feature_engineering(raw_data.copy())

        # Add an interactive filter
        st.sidebar.subheader("EDA Filters")
        amount_range = st.sidebar.slider(
            "Filter by Transaction Amount:",
            min_value=float(raw_data['amount'].min()),
            max_value=float(raw_data['amount'].max()),
            value=(float(raw_data['amount'].min()), float(raw_data['amount'].max()))
        )

        # Filter data based on slider
        filtered_df = df_feat[
            (df_feat['amount'] >= amount_range[0]) &
            (df_feat['amount'] <= amount_range[1])
            ]

        st.write(f"Showing {len(filtered_df)} transactions out of {len(df_feat)}")

        st.subheader("Time-Series Feature Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Viz 4: Time Since Last Transaction (Plotly)
            st.write("##### Time Since Last Transaction (Log Scale)")
            filtered_df['log_time_since'] = np.log1p(filtered_df['time_since_last_tx_seconds'])
            fig4 = px.histogram(filtered_df.dropna(), x='log_time_since', color='is_fraud',
                                title="Log(Time Since Last Tx) by Fraud Status", nbins=50, marginal='box')
            fig4.update_layout(template='plotly_dark')
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            # Viz 5: Transaction Count in Last 1H (Plotly)
            st.write("##### Transaction Count in Last 1 Hour")
            filtered_df['tx_count_last_1h_clipped'] = filtered_df['tx_count_last_1h'].clip(upper=10)
            fig5 = px.histogram(filtered_df.dropna(), x='tx_count_last_1h_clipped', color='is_fraud',
                                title="Tx Count in Last 1H (Clipped at 10)", nbins=11)
            fig5.update_layout(template='plotly_dark')
            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Categorical Feature Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Viz 6: Fraud by Device (Seaborn)
            st.write("##### Fraud by Device Type")
            fig, ax = plt.subplots()
            sns.countplot(data=filtered_df, x='device', hue='is_fraud', ax=ax, palette='pastel')
            ax.set_title("Fraud by Device Type")
            st.pyplot(fig)

        with col2:
            # Viz 7: Fraud by Hour of Day (Seaborn)
            st.write("##### Fraud by Hour of Day")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(data=filtered_df, x='hour_of_day', hue='is_fraud', ax=ax, palette='twilight')
            ax.set_title("Fraud by Hour of Day")
            st.pyplot(fig)

        st.subheader("Numerical Feature Correlation")
        # Viz 8: Correlation Heatmap (Seaborn)
        st.write("##### Correlation Heatmap")
        num_cols = ['amount', 'time_since_last_tx_seconds', 'tx_count_last_1h', 'hour_of_day', 'is_fraud']
        corr = filtered_df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', ax=ax)
        st.pyplot(fig)


    #Fraud Model
    elif page == "Fraud Model & Simulation":
        st.header("Fraud Detection Model & Live Simulation")

        # Load feature-engineered data
        df_model = advanced_feature_engineering(raw_data.copy())

        # Train model
        pipeline, report, auc_score, cm, feature_names = train_model(df_model.copy())

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "Random Forest")
            st.metric("ROC AUC Score", f"{auc_score:.4f}")
            st.write("##### Classification Report")
            st.text(report)

        with col2:
            # Viz 9: Confusion Matrix (Seaborn)
            st.write("##### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Legitimate', 'Fraud'],
                        yticklabels=['Legitimate', 'Fraud'])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Viz 10: Feature Importance (Plotly)
        st.write("##### Model Feature Importance")

        try:
            cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
                input_features=['location', 'device', 'day_of_week'])
            num_features = ['amount', 'hour_of_day', 'time_since_last_tx_seconds', 'tx_count_last_1h']

            all_feature_names = np.concatenate([num_features, cat_features])

            importances = pipeline.named_steps['classifier'].feature_importances_

            feature_importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(15)  # Show top 15

            fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                             title="Top 15 Most Important Features")
            fig_imp.update_layout(template='plotly_dark')
            st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.write(f"Could not plot feature importance: {e}")

        st.sidebar.subheader("Live Fraud Simulation")
        st.sidebar.write("Test the model with a new transaction:")

        # Create input fields
        amount = st.sidebar.number_input("Amount:", min_value=0.0, value=150.0)
        location = st.sidebar.selectbox("Location:", options=raw_data['location'].unique())
        device = st.sidebar.selectbox("Device:", options=raw_data['device'].unique())
        hour_of_day = st.sidebar.slider("Hour of Day:", 0, 23, int(time.strftime("%H")))
        day_of_week = st.sidebar.slider("Day of Week (0=Sun):", 0, 7, int(time.strftime("%w")) - 0)
        time_since_last_tx = st.sidebar.number_input("Seconds Since Last Tx:", min_value=0.0, value=3600.0)
        tx_count_last_1h = st.sidebar.number_input("Transactions in Last Hour:", min_value=0, value=1)

        if st.sidebar.button("Predict Fraud", use_container_width=True):
            # Create a DataFrame from the inputs
            input_data = pd.DataFrame({
                'amount': [amount],
                'location': [location],
                'device': [device],
                'hour_of_day': [hour_of_day],
                'day_of_week': [day_of_week],
                'time_since_last_tx_seconds': [time_since_last_tx],
                'tx_count_last_1h': [tx_count_last_1h]
            })

            input_data = input_data[feature_names.drop(['is_fraud', 'timestamp'], errors='ignore')]

            prediction = pipeline.predict(input_data)[0]
            prediction_proba = pipeline.predict_proba(input_data)[0]

            prob_fraud = prediction_proba[1]

            st.sidebar.subheader("Prediction Result:")
            if prediction == 1:
                st.sidebar.error(f"**FRAUD** (Probability: {prob_fraud:.2%})", icon="🚨")
            else:
                st.sidebar.success(f"**Legitimate** (Fraud Prob: {prob_fraud:.2%})", icon="✅")
