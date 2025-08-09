
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_class_distribution(df, target_column, save_path='plots/class_distribution.png'):
    """Plot the distribution of fraud vs non-fraud cases"""
    plt.figure(figsize=(10, 6))
    
    # Get value counts
    counts = df[target_column].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    
    # Create barplot
    ax = sns.barplot(x=counts.index, y=counts.values)
    
    # Add value labels and percentages
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        ax.text(i, count/2, f'Count: {count}\n{percentage}%', 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Distribution of Fraud vs Normal Transactions', pad=20)
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Number of Transactions')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_distributions(df, save_path='plots/feature_distributions.png'):
    """Plot distributions of numerical features with fraud overlay"""
    numerical_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                     if col != 'fraud']
    num_features = len(numerical_cols)
    
    n_cols = 2
    n_rows = (num_features + 1) // 2
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, feature in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, n_cols, idx)
        
        if df[feature].var() > 0:
            sns.kdeplot(data=df, x=feature, hue='fraud', warn_singular=False)
        else:
            plt.hist(df[feature], bins=20, label=['Normal', 'Fraud'])
            plt.legend()
        
        plt.title(f'Distribution of {feature}')
    
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_correlation_heatmap(df, save_path='plots/correlation_heatmap.png'):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=(12, 8))
    
    corr = df.corr()
    
    sns.heatmap(corr, 
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True)
    
    plt.title('Feature Correlation Matrix', pad=20)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def compare_real_synthetic(real_df, synthetic_df, feature, save_path):
    """Compare distributions between real and synthetic data"""
    fig = plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    if real_df[feature].var() > 0:
        sns.kdeplot(data=real_df, x=feature, hue='fraud', warn_singular=False)
    else:
        plt.hist(real_df[feature], bins=20)
    plt.title(f'Real Data - {feature}')
    
    plt.subplot(1, 2, 2)
    if synthetic_df[feature].var() > 0:
        sns.kdeplot(data=synthetic_df, x=feature, hue='fraud', warn_singular=False)
    else:
        plt.hist(synthetic_df[feature], bins=20)
    plt.title(f'Synthetic Data - {feature}')
    
    plt.subplots_adjust(wspace=0.3, top=0.9)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_visualizations(df, target_column='fraud'):
    """Create and save all visualizations"""
    os.makedirs('plots', exist_ok=True)
    
    print("Generating visualizations...")
    
    try:
        # 1. Class distribution
        plot_class_distribution(df, target_column)
        print("✓ Created class distribution plot")
        
        # 2. Feature distributions
        plot_feature_distributions(df)
        print("✓ Created feature distribution plots")
        
        # 3. Correlation heatmap
        plot_correlation_heatmap(df)
        print("✓ Created correlation heatmap")
        
        # 4. Compare with synthetic data if available
        synthetic_path = os.path.join('data', 'synthetic_fraud_data.csv')
        if os.path.exists(synthetic_path):
            print("Generating synthetic data comparisons...")
            synthetic_df = pd.read_csv(synthetic_path)
            
            for feature in df.columns:
                if feature != target_column:
                    compare_real_synthetic(
                        df, 
                        synthetic_df, 
                        feature,
                        f'plots/synthetic_comparison_{feature}.png'
                    )
            print("✓ Created synthetic data comparisons")
        
        print("\nAll visualizations have been saved to the 'plots' directory")
    
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise