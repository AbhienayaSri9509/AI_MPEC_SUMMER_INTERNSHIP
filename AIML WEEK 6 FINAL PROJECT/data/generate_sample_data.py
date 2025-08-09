import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.1):
    """
    Generate a synthetic fraud dataset with the following features:
    - amount: transaction amount
    - hour: hour of transaction
    - age: customer age
    - distance: distance from usual location
    - fraud: target variable (0: normal, 1: fraud)
    """
    # Generate binary classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_redundant=0,
        n_informative=4,
        random_state=42,
        weights=[1-fraud_ratio]
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['amount', 'hour', 'age', 'distance'])
    
    # Scale and transform features to meaningful ranges
    df['amount'] = np.abs(df['amount'] * 1000 + 500)  # Transaction amounts between 0 and ~2000
    df['hour'] = np.abs((df['hour'] * 12 + 12) % 24)  # Hours between 0 and 24
    df['age'] = np.abs(df['age'] * 20 + 35)  # Age between 18 and ~70
    df['distance'] = np.abs(df['distance'] * 50)  # Distance in km
    
    # Add fraud labels
    df['fraud'] = y
    
    return df

def main():
    # Generate dataset
    df = generate_fraud_dataset()
    
    # Save to CSV
    output_path = 'data/fraud_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated sample fraud dataset with {len(df)} records")
    print(f"Saved to: {output_path}")
    
    # Display class distribution
    fraud_count = df['fraud'].sum()
    print(f"\nClass distribution:")
    print(f"Normal transactions: {len(df) - fraud_count}")
    print(f"Fraudulent transactions: {fraud_count}")

if __name__ == "__main__":
    main()