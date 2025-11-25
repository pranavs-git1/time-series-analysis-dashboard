import matplotlib

matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """Loads and performs initial timestamp conversion."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def feature_engineering(df):
    """Creates new features from the timestamp."""
    print("Performing feature engineering...")
    df_feat = df.copy()
    df_feat['hour_of_day'] = df_feat['timestamp'].dt.hour
    df_feat['day_of_week'] = df_feat['timestamp'].dt.dayofweek
    df_feat = df_feat.drop(['timestamp', 'transaction_id', 'customer_id'], axis=1)
    return df_feat

def plot_eda(df):
    """Generates and saves EDA plots using Matplotlib and Seaborn."""
    print("Generating EDA plots...")

    sns.set_style("darkgrid")

    # 1.plot transaction distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='is_fraud', data=df, palette='pastel')
    plt.title('Distribution of Fraud vs. Legitimate Transactions')
    plt.xlabel('Is Fraud? (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig('eda_fraud_distribution.png')
    print("Saved eda_fraud_distribution.png")

    # 2.plot amount distribution by fraud status
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df,
        x='amount',
        hue='is_fraud',
        kde=True,
        bins=100,
        palette='muted'
    )
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.xlabel('Amount')
    plt.xlim(0, 3000) 
    plt.savefig('eda_amount_distribution.png')
    print("Saved eda_amount_distribution.png")

    # 3.plot fraud by hour of day
    plt.figure(figsize=(12, 6))
    sns.countplot(x='hour_of_day', data=df, hue='is_fraud', palette='twilight')
    plt.title('Transaction Counts by Hour of Day')
    plt.savefig('eda_hourly_fraud.png')
    print("Saved eda_hourly_fraud.png")

    plt.close('all')


def train_model(df):
    """Preprocesses data, trains, and evaluates a model."""
    print("Starting model training pipeline...")

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    categorical_features = ['location', 'device', 'day_of_week']
    numerical_features = ['amount', 'hour_of_day']

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',  
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Fitting model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- ROC AUC Score ---")
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    print(f"{roc_auc_score(y_test, y_probs):.4f}")

    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('model_confusion_matrix.png')
    print("Saved model_confusion_matrix.png")


if __name__ == "__main__":
    data = load_data('transactions.csv')

    data_feat = feature_engineering(data)

    plot_eda(data_feat)

    train_model(data_feat)

    print("\nAnalysis complete. Check for .png files in the directory.")
