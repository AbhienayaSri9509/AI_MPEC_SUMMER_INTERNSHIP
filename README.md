# 🎓 AI_MPEC Summer Internship Projects

Welcome to the AI_MPEC Summer Internship repository!

This repository showcases five weekly mini-projects and a Week 6 final project, completed as part of the internship.
Each project demonstrates practical applications of data analysis, clustering, and machine learning in various domains.

---

## 📅 WEEK 1: Healthcare Cost Analysis by Country

### 📌 Objective:
To analyze global healthcare expenditure and uncover trends across different countries.

### 📊 Dataset:
- `Life Expectancy Data.xls` (from WHO / World Bank)

### 🛠️ Techniques Used:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Correlation analysis between life expectancy and health spending
- Visualization using matplotlib/seaborn

### 📌 Outcome:
Identified countries with the highest and lowest healthcare spending, discovered how it relates to life expectancy, and visualized key insights using graphs.

---

## 📅 WEEK 2: Cost of Living vs Average Salary

### 📌 Objective:
To compare the cost of living and average income across major cities and evaluate where living is affordable.

### 📊 Dataset:
- `cost_living_data.csv`

### 🛠️ Techniques Used:
- Data preprocessing
- Normalization and indexing of cost/salary
- Scatter plots and city-level ranking
- Country-wise aggregation

### 📌 Outcome:
Created a ranked comparison of cities based on salary-to-cost ratio, helping identify affordable living locations worldwide.

---

## 📅 WEEK 3: Segmenting Credit Card Users

### 📌 Objective:
To apply unsupervised learning to group credit card users based on their spending behavior and profile.

### 📊 Dataset:
- `Credit Card Dataset.csv`

### 🛠️ Techniques Used:
- K-Means clustering
- Principal Component Analysis (PCA) for dimensionality reduction
- Customer segmentation based on spending categories
- Visualization of clusters

### 📌 Outcome:
Successfully segmented users into meaningful groups (e.g., high spenders, low balance holders, etc.), which can help in targeted marketing and financial strategy planning.

---

## 📅 WEEK 4: Movie User Taste Clustering

### 📌 Objective:
To group movie users based on their preferences and viewing history.

### 📊 Dataset:
- MovieLens or custom movie-user preference matrix

### 🛠️ Techniques Used:
- Cosine similarity
- K-Means clustering
- User-movie rating matrix
- Cluster visualization

### 📌 Outcome:
Users were clustered into taste groups (e.g., Action Lovers, Romantic Movie Fans, Comedy Watchers), enabling personalized recommendation strategies.

---

## 📅 WEEK 5: Alcohol Speech Detection Using Audio Classification

### 📌 Objective:
To classify whether a person is under the influence of alcohol based on their speech patterns using audio feature extraction and supervised machine learning techniques.

### 📊 Dataset:
- `alcohol_speech_data.csv` or extracted features from real-time audio recordings

### 🛠️ Techniques Used:
- **Audio Preprocessing:**
  - Noise filtering and normalization
  - Audio segmentation and feature extraction using:
    - MFCC (Mel-frequency cepstral coefficients)
    - Chroma Frequencies
    - Zero-Crossing Rate
    - Root Mean Square Energy (RMSE)
- **Model Training:**
  - Train-test split
  - Algorithms: Random Forest, SVM, Logistic Regression
  - Evaluation using accuracy, F1-score, and confusion matrix
- **Visualization:**
  - MFCC spectrograms
  - Confusion matrix plots
  - Feature importance charts

### 📌 Outcome:
Built a reliable classifier to distinguish between sober and intoxicated speech patterns with strong accuracy. Demonstrated real-world application potential for safety monitoring, healthcare assessments, and DUI prevention systems.

---

#### 📖WEEK 6 FINAL PROJECT : Fraud Detection with CTGAN

## 📌 Overview

This project demonstrates how CTGAN (Conditional Tabular Generative Adversarial Network) can be used to generate synthetic transaction data for fraud detection tasks.
CTGAN helps in handling imbalanced datasets by creating realistic synthetic fraud samples, improving model training for rare-event classification problems.

## 🎯 Objectives

Generate synthetic fraudulent transaction data to balance the dataset.

Improve fraud detection model performance using augmented data.

Showcase a complete data preprocessing → generation → model training → evaluation pipeline.

## 🛠️ Technologies Used

Python (>=3.8)

Pandas, NumPy – Data processing

scikit-learn – ML models & evaluation

ctgan – Synthetic data generation

Matplotlib, Seaborn – Data visualization


## 📊 Dataset

We use a credit card transactions dataset (e.g., Kaggle Credit Card Fraud Detection Dataset), which is highly imbalanced:

Legitimate transactions: ~99.8%

Fraudulent transactions: ~0.2%

## 🔄 Workflow
Load and preprocess transaction data.

Train CTGAN on fraud-labeled data to generate synthetic samples.

Augment original dataset with synthetic fraud transactions.

Train ML models (Random Forest, XGBoost, Logistic Regression) on the balanced dataset.

Evaluate performance using Precision, Recall, F1-score, and ROC-AUC.

## 📈 Results

Model	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.85	0.72	0.78	0.95
Random Forest	0.91	0.88	0.89	0.98
XGBoost	0.93	0.90	0.91	0.99

CTGAN-based data augmentation significantly improves recall while maintaining high precision.

**▶️ How to Run**

**1. Clone the repository**

cd fraud-ctgan

**2. Install dependencies**
pip install -r requirements.txt

**3. Run preprocessing**
python src/preprocess.py

**4. Generate synthetic data with CTGAN**
python src/generate_ctgan.py

**5. Train fraud detection model**
python src/train_model.py

## 📌 Future Improvements
Experiment with TVAE (Tabular VAE) for comparison.

Hyperparameter tuning for CTGAN.

Deploy fraud detection model as an API.

Test on multiple fraud datasets. 
 
## jupyter link:
file:///C:/Users/HP/Downloads/Untitled.ipynb%20-%20JupyterLab.html

## drive link of the project folder :
 https://drive.google.com/drive/folders/1GIMs2DOi7VI_74CIv3AsYWCaNcQFqgJu?usp=drive_link 

 
## 💼 Intern Skill Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (Clustering, PCA, Classification)
- Librosa & Pydub for audio signal processing
- Jupyter Notebooks & Google Colab
- Data storytelling and visualization

---

## 📬 Contact

**Intern Name:** Abhienaya Sri  
**GitHub:** [AbhienayaSri9509](https://github.com/AbhienayaSri9509)  
**Email:** abhienayasris@gmail.com  

**AI MPEC SUMMER INTERNSHIP GITHUB LINK:** [AI_MPEC_SUMMER_INTERNSHIP](https://github.com/AbhienayaSri9509/AI_MPEC_SUMMER_INTERNSHIP/tree/master)

---
