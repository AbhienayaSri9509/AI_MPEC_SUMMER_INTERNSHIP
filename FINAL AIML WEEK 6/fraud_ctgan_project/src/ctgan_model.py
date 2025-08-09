
from ctgan import CTGAN
import pandas as pd
import numpy as np
import os

class FraudCTGAN:
    def __init__(self, epochs=50):
        """Initialize CTGAN model with optimized parameters"""
        self.ctgan = CTGAN(
            epochs=epochs,
            batch_size=500,
            generator_dim=(128, 128, 128),
            discriminator_dim=(128, 128, 128),
            verbose=True,
            # Remove multiprocessing parameters to avoid WMIC issues
            cuda=False  # Disable GPU usage for stability
        )
        
    def train(self, data):
        """Train the CTGAN model with progress tracking"""
        try:
            print("Starting CTGAN training...")
            print(f"Training on {len(data)} samples")
            self.ctgan.fit(data)
            print("Training completed successfully")
            return True
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def generate_samples(self, n_samples):
        """Generate synthetic samples"""
        try:
            print(f"Generating {n_samples} synthetic samples...")
            synthetic_data = self.ctgan.sample(n_samples)
            print("Sample generation completed")
            return synthetic_data
        except Exception as e:
            print(f"Error generating samples: {str(e)}")
            return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.ctgan.save(filepath)
            print(f"Model saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model(filepath):
        """Load a trained model"""
        try:
            model = CTGAN.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None