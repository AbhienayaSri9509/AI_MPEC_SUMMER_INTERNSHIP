import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load the fraud dataset"""
        df = pd.read_csv(filepath)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for CTGAN"""
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def split_data(self, df, target_column, test_size=0.2):
        """Split data into train and test sets"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)