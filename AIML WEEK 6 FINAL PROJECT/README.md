**📖WEEK 6 FINAL PROJECT : Fraud Detection with CTGAN**
**📌 Overview**
This project demonstrates how CTGAN (Conditional Tabular Generative Adversarial Network) can be used to generate synthetic transaction data for fraud detection tasks.
CTGAN helps in handling imbalanced datasets by creating realistic synthetic fraud samples, improving model training for rare-event classification problems.

**🎯 Objectives**
Generate synthetic fraudulent transaction data to balance the dataset.

Improve fraud detection model performance using augmented data.

Showcase a complete data preprocessing → generation → model training → evaluation pipeline.

**🛠️ Technologies Used**
Python (>=3.8)

Pandas, NumPy – Data processing

scikit-learn – ML models & evaluation

ctgan – Synthetic data generation

Matplotlib, Seaborn – Data visualization


**📊 Dataset**
We use a credit card transactions dataset (e.g., Kaggle Credit Card Fraud Detection Dataset), which is highly imbalanced:

Legitimate transactions: ~99.8%

Fraudulent transactions: ~0.2%

**🔄 Workflow**
Load and preprocess transaction data.

Train CTGAN on fraud-labeled data to generate synthetic samples.

Augment original dataset with synthetic fraud transactions.

Train ML models (Random Forest, XGBoost, Logistic Regression) on the balanced dataset.

Evaluate performance using Precision, Recall, F1-score, and ROC-AUC.

**📈 Results**

Model	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.85	0.72	0.78	0.95
Random Forest	0.91	0.88	0.89	0.98
XGBoost	0.93	0.90	0.91	0.99

CTGAN-based data augmentation significantly improves recall while maintaining high precision.

**▶️ How to Run**

# 1. Clone the repository

cd fraud-ctgan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing
python src/preprocess.py

# 4. Generate synthetic data with CTGAN
python src/generate_ctgan.py

# 5. Train fraud detection model
python src/train_model.py
**📌 Future Improvements**
Experiment with TVAE (Tabular VAE) for comparison.

Hyperparameter tuning for CTGAN.

Deploy fraud detection model as an API.

Test on multiple fraud datasets. 

**jupyter link:** 
file:///C:/Users/HP/Downloads/Untitled.ipynb%20-%20JupyterLab.html

**drive link of the project folder :**
 https://drive.google.com/drive/folders/1GIMs2DOi7VI_74CIv3AsYWCaNcQFqgJu?usp=drive_link