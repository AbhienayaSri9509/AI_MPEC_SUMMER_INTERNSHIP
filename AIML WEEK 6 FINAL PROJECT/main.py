
from visualization import create_visualizations
from data_preprocessing import DataPreprocessor
from generate_sample_data import generate_fraud_dataset
from train import main as train_model
import os

def main():
    
    os.makedirs('data', exist_ok=True)
    
    
    data_path = 'data/fraud_data.csv'
    if not os.path.exists(data_path):
        df = generate_fraud_dataset()
        df.to_csv(data_path, index=False)
        print(f"Generated sample dataset and saved to {data_path}")
    
    # Load data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, target_column='fraud')
    
    # Run training
    print("\nStarting model training...")
    train_model()

if __name__ == "__main__":
    main()