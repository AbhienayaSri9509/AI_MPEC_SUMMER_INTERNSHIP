

**📅 WEEK 6 (Final Project) – Fraud Detection with CTGAN**
**📌 Objective**
Use CTGAN (Conditional Tabular GAN) to generate synthetic fraudulent transaction data, solve dataset imbalance issues, and improve rare-event fraud detection performance.

**📊 Dataset**
Source: Kaggle Credit Card Fraud Detection Dataset

**Class distribution:**

Legitimate transactions: ~99.8%

Fraudulent transactions: ~0.2%

**🛠️ Technologies Used**
Python (>=3.8)

Pandas, NumPy – Data preprocessing

scikit-learn – Model training & evaluation

CTGAN – Synthetic data generation

Matplotlib, Seaborn – Visualization

**📂 Project Structure**

fraud-ctgan/
│── data/
│   ├── original_transactions.csv     # Raw dataset
│   ├── synthetic_transactions.csv    # CTGAN-generated data
│── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── ctgan_training.ipynb
│   ├── model_training.ipynb
│── src/
│   ├── preprocess.py
│   ├── generate_ctgan.py
│   ├── train_model.py
│── requirements.txt
│── README.md

**🔄 Workflow**
Data Preprocessing – Clean and split dataset into fraud & non-fraud transactions.

CTGAN Training – Train model using only fraudulent transactions.

Synthetic Data Generation – Generate new fraud samples.

Dataset Augmentation – Merge synthetic data with original dataset to balance classes.

Model Training – Train fraud detection models (Random Forest, XGBoost, Logistic Regression).

Evaluation – Compare performance before and after data augmentation.

**📈 Results**
Model	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.85	0.72	0.78	0.95
Random Forest	0.91	0.88	0.89	0.98
XGBoost	0.93	0.90	0.91	0.99

Insight:
CTGAN-based augmentation improved recall significantly without major precision loss, making it better at catching fraudulent cases.

**▶️ How to Run**


# 1. Clone repository
git clone https://github.com/yourusername/fraud-ctgan.git
cd fraud-ctgan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess dataset
python src/preprocess.py

# 4. Train CTGAN & generate synthetic data
python src/generate_ctgan.py

# 5. Train fraud detection models
python src/train_model.py
**📌 Future Improvements**
Compare CTGAN with TVAE (Tabular VAE).

Hyperparameter tuning for CTGAN.

Deploy as REST API for real-time fraud detection.

Test across multiple fraud datasets for robustness.

**JUPYTER LAB** : http://localhost:8890/lab

