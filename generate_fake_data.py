import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration
N_CUSTOMERS = 1000
N_TRANSACTIONS = 20000
FRAUD_RATE = 0.10

LOCATIONS = ['New York', 'London', 'Tokyo', 'Mumbai', 'San Francisco', 'Unknown', 'Paris']
DEVICES = ['Mobile', 'Desktop', 'Tablet', 'Emulator', 'POS']

customer_ids = [f'C_{i:05d}' for i in range(1, N_CUSTOMERS + 1)]


def create_fake_transactions(n):
    """Generates a list of fake transaction dictionaries."""
    data = []
    start_date = datetime(2024, 1, 1)

    print(f"Generating {n} transactions...")

    for i in range(n):
        cust_id = random.choice(customer_ids)
        timestamp = start_date + timedelta(days=random.uniform(0, 364), hours=random.uniform(0, 23),
                                           minutes=random.uniform(0, 59))

        is_fraud = False
        amount = round(random.uniform(5.0, 300.0), 2)
        location = random.choice(LOCATIONS[:-1]) 
        device = random.choice(DEVICES[:-2])  

        if random.random() < FRAUD_RATE:
            is_fraud = True

            fraud_type = random.choice(['high_amount', 'unknown_location', 'emulator'])

            if fraud_type == 'high_amount':
                amount = round(random.uniform(1000.0, 5000.0), 2)
            elif fraud_type == 'unknown_location':
                location = 'Unknown'
                amount = round(random.uniform(100.0, 1500.0), 2)
            else:  # emulator
                device = 'Emulator'
                amount = round(random.uniform(50.0, 800.0), 2)

        data.append({
            'transaction_id': f'T_{1000000 + i}',
            'customer_id': cust_id,
            'timestamp': timestamp,
            'amount': amount,
            'location': location,
            'device': device,
            'is_fraud': int(is_fraud)
        })

    print("Generation complete.")
    return data


if __name__ == "__main__":
    transactions_data = create_fake_transactions(N_TRANSACTIONS)

    df = pd.DataFrame(transactions_data)

    output_file = 'transactions.csv'
    df.to_csv(output_file, index=False)

    print(f"Successfully generated and saved {len(df)} transactions to {output_file}")
    print("\nData preview:")
    print(df.head())
    print(f"\nFraud vs. Legitimate transactions:\n{df['is_fraud'].value_counts()}")
