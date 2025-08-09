
import os
import sys
from data_preprocessing import DataPreprocessor
from ctgan_model import FraudCTGAN
import warnings

def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        data_path = os.path.join('data', 'fraud_data.csv')
        
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            print("Please run generate_sample_data.py first")
            sys.exit(1)
            
        print("Loading data...")
        df = preprocessor.load_data(data_path)
        processed_df = preprocessor.preprocess_data(df)
        
        print("Training CTGAN model...")
        print("(This may take a while, using single CPU core for stability)")
        ctgan = FraudCTGAN(epochs=50)  # Reduced epochs for testing
        ctgan.train(processed_df)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the trained model
        model_path = os.path.join('models', 'fraud_ctgan.pkl')
        ctgan.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Generate synthetic samples
        print("Generating synthetic samples...")
        synthetic_samples = ctgan.generate_samples(1000)
        synthetic_path = os.path.join('data', 'synthetic_fraud_data.csv')
        synthetic_samples.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to {synthetic_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()