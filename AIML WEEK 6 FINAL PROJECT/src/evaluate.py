
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
import json

class SyntheticDataEvaluator:
    def __init__(self, real_data_path, synthetic_data_path):
        """Initialize evaluator with paths to real and synthetic data"""
        # Load data with proper data types
        self.real_df = pd.read_csv(real_data_path)
        self.synthetic_df = pd.read_csv(synthetic_data_path)
        
        # Ensure consistent data types between real and synthetic data
        for column in self.real_df.columns:
            dtype = self.real_df[column].dtype
            self.synthetic_df[column] = self.synthetic_df[column].astype(dtype)
        
        self.metrics = {}

    def calculate_statistical_metrics(self):
        """Calculate statistical similarity metrics between real and synthetic data"""
        metrics = {}
        
        for column in self.real_df.columns:
            if self.real_df[column].dtype in ['int64', 'float64']:
                # Kolmogorov-Smirnov test
                ks_statistic, p_value = ks_2samp(
                    self.real_df[column], 
                    self.synthetic_df[column]
                )
                
                metrics[column] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'real_mean': self.real_df[column].mean(),
                    'synthetic_mean': self.synthetic_df[column].mean(),
                    'real_std': self.real_df[column].std(),
                    'synthetic_std': self.synthetic_df[column].std(),
                    'real_median': self.real_df[column].median(),
                    'synthetic_median': self.synthetic_df[column].median()
                }
        
        self.metrics['statistical'] = metrics
        return metrics

    def evaluate_privacy(self):
        """Enhanced privacy evaluation"""
        # Convert all columns to same dtype for comparison
        real_compare = self.real_df.copy()
        synth_compare = self.synthetic_df.copy()
        
        # Ensure all numeric columns are float for comparison
        for column in real_compare.columns:
            if real_compare[column].dtype in ['int64', 'float64']:
                real_compare[column] = real_compare[column].astype(float)
                synth_compare[column] = synth_compare[column].astype(float)
        
        # Check for exact duplicates
        exact_duplicates = len(
            pd.merge(real_compare, synth_compare, how='inner')
        )
        
        # Calculate privacy metrics
        self.metrics['privacy'] = {
            'exact_duplicates': exact_duplicates,
            'duplicate_percentage': (exact_duplicates / len(self.real_df)) * 100
        }
        return self.metrics['privacy']

    
    def evaluate_ml_performance(self):
        """Compare ML model performance on real vs synthetic data"""
        def train_and_evaluate(X_train, X_test, y_train, y_test):
            try:
                # Initialize and train the model
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=1  # Set to 1 to avoid multiprocessing issues
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]  # Get probability for positive class
                
                # Calculate metrics for binary classification
                metrics_dict = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'roc_auc': roc_auc_score(y_test, y_prob)
                }
                
                return metrics_dict
                
            except Exception as e:
                print(f"Warning: Error in ML evaluation: {str(e)}")
                return {
                    'accuracy': None,
                    'classification_report': None,
                    'roc_auc': None
                }

        try:
            # Prepare features and target
            features = [col for col in self.real_df.columns if col != 'fraud']
            
            # Real data performance
            print("Evaluating real data performance...")
            X_real = self.real_df[features]
            y_real = self.real_df['fraud']
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_real
            )
            real_performance = train_and_evaluate(
                X_train_real, X_test_real, 
                y_train_real, y_test_real
            )

            # Synthetic data performance
            print("Evaluating synthetic data performance...")
            X_synthetic = self.synthetic_df[features]
            y_synthetic = self.synthetic_df['fraud']
            X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
                X_synthetic, y_synthetic, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_synthetic
            )
            synthetic_performance = train_and_evaluate(
                X_train_synth, X_test_synth, 
                y_train_synth, y_test_synth
            )

            # Store results
            self.metrics['ml_performance'] = {
                'real_data': real_performance,
                'synthetic_data': synthetic_performance
            }
            
            print("ML performance evaluation completed successfully")
            return self.metrics['ml_performance']
            
        except Exception as e:
            print(f"Error in ML performance evaluation: {str(e)}")
            self.metrics['ml_performance'] = {
                'real_data': {'accuracy': None, 'classification_report': None, 'roc_auc': None},
                'synthetic_data': {'accuracy': None, 'classification_report': None, 'roc_auc': None}
            }
            return self.metrics['ml_performance']
           
      

    def plot_distributions(self):
        """Plot distribution comparisons for each feature"""
        os.makedirs('plots/evaluation', exist_ok=True)
        
        for column in self.real_df.columns:
            if self.real_df[column].dtype in ['int64', 'float64']:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                sns.histplot(data=self.real_df, x=column, stat='density', label='Real')
                sns.histplot(data=self.synthetic_df, x=column, stat='density', label='Synthetic', alpha=0.6)
                plt.title(f'Density Distribution - {column}')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                sns.boxplot(data=pd.DataFrame({
                    'Real': self.real_df[column],
                    'Synthetic': self.synthetic_df[column]
                }).melt(), x='variable', y='value')
                plt.title(f'Box Plot - {column}')
                
                plt.tight_layout()
                plt.savefig(f'plots/evaluation/dist_comparison_{column}.png')
                plt.close()

    def generate_report(self, save_path='evaluation_report.txt', json_path='evaluation_metrics.json'):
        """Generate comprehensive evaluation report"""
        # Calculate all metrics
        self.calculate_statistical_metrics()
        self.evaluate_privacy()
        self.evaluate_ml_performance()
        
        # Save metrics as JSON
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Create formatted report
        with open(save_path, 'w') as f:
            f.write("Synthetic Data Evaluation Report\n")
            f.write("===============================\n\n")
            
            # Statistical metrics
            f.write("1. Statistical Metrics\n")
            f.write("--------------------\n")
            for column, metrics in self.metrics['statistical'].items():
                f.write(f"\n{column}:\n")
                f.write("  Real Data Stats:\n")
                f.write(f"    Mean: {metrics['real_mean']:.4f}\n")
                f.write(f"    Median: {metrics['real_median']:.4f}\n")
                f.write(f"    Std: {metrics['real_std']:.4f}\n")
                f.write("\n  Synthetic Data Stats:\n")
                f.write(f"    Mean: {metrics['synthetic_mean']:.4f}\n")
                f.write(f"    Median: {metrics['synthetic_median']:.4f}\n")
                f.write(f"    Std: {metrics['synthetic_std']:.4f}\n")
                f.write("\n  Similarity Tests:\n")
                f.write(f"    KS-test statistic: {metrics['ks_statistic']:.4f}\n")
                f.write(f"    KS-test p-value: {metrics['p_value']:.4f}\n")
                f.write("\n")
            
            # Privacy metrics
            f.write("\n2. Privacy Metrics\n")
            f.write("----------------\n")
            for metric, value in self.metrics['privacy'].items():
                if value is not None:
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: N/A\n")
            
            # ML Performance comparison
            f.write("\n3. Machine Learning Performance Comparison\n")
            f.write("--------------------------------------\n")
            
            # Real data performance
            f.write("\nReal Data Performance:\n")
            real_perf = self.metrics['ml_performance']['real_data']
            if real_perf['accuracy'] is not None:
                f.write(f"Accuracy: {real_perf['accuracy']:.4f}\n")
                f.write(f"ROC-AUC Score: {real_perf['roc_auc']:.4f}\n")
            else:
                f.write("Metrics not available due to evaluation error\n")
            
            # Synthetic data performance
            f.write("\nSynthetic Data Performance:\n")
            synth_perf = self.metrics['ml_performance']['synthetic_data']
            if synth_perf['accuracy'] is not None:
                f.write(f"Accuracy: {synth_perf['accuracy']:.4f}\n")
                f.write(f"ROC-AUC Score: {synth_perf['roc_auc']:.4f}\n")
            else:
                f.write("Metrics not available due to evaluation error\n")

def main():
    # Paths to data
    real_data_path = 'data/fraud_data.csv'
    synthetic_data_path = 'data/synthetic_fraud_data.csv'
    
    try:
        # Create evaluator
        evaluator = SyntheticDataEvaluator(real_data_path, synthetic_data_path)
        
        # Generate visualizations
        print("Generating distribution comparisons...")
        evaluator.plot_distributions()
        
        # Generate comprehensive report
        print("Generating evaluation report...")
        evaluator.generate_report()
        
        print("Evaluation complete. Check 'evaluation_report.txt' for detailed metrics.")
        print("Visualization plots are saved in 'plots/evaluation' directory.")
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()