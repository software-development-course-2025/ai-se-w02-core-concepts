# ==============================================================================
# SDG 3: Health and Well-being - Diabetes Risk Prediction (Supervised Learning)
#
# This script implements a Supervised Classification model (Logistic Regression)
# to predict the risk of diabetes using the Pima Indians Diabetes Dataset.
#
# Deliverables Covered: Technical Implementation (40%), ML Approach, Evaluation.
# Tools Used: Pandas, NumPy, Scikit-learn, Matplotlib.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Setup and Data Loading (SDG and Dataset Requirement)
# ------------------------------------------------------------------------------

import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: The 'diabetes.csv' file containing the Pima Indians Diabetes Dataset
# must be uploaded to your working environment (Colab/Jupyter directory).
# Target variable is 'Outcome' (1=Diabetic, 0=Non-Diabetic).

try:
    # Load the dataset
    df = pd.read_csv('diabetes.csv')
    print("Data loaded successfully. First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("\n[CRITICAL ERROR] 'diabetes.csv' not found.")
    print("Please ensure the file is in the same directory as the script.")
    sys.exit(1)
    # Exit or use sample data if necessary

# ------------------------------------------------------------------------------
# 2. Data Preprocessing (Cleaning and Splitting)
# ------------------------------------------------------------------------------

# The dataset contains zero values (0) which are physiologically invalid for
# certain features (e.g., Glucose, BMI, BloodPressure).
# We replace these invalid zeros with the column's mean (a simple imputation strategy).
cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
df.fillna(df.mean(), inplace=True) # Imputation

# Define Features (X) and Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into Training and Testing sets (80% / 20%)
# The test set simulates 'unseen data' for final model evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")

# Data Standardization (Scaling)
# Scaling features is essential for Logistic Regression and Neural Networks.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------------------------
# 3. Model Training (Supervised Learning Approach)
# ------------------------------------------------------------------------------

# Initialize and train the Logistic Regression model (a robust classification algorithm).
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)


# ------------------------------------------------------------------------------
# 4. Model Evaluation (Required Metrics and Visualization)
# ------------------------------------------------------------------------------

print("\n--- Model Evaluation Report ---")
# Use required classification metrics (Accuracy, Precision, Recall, F1-Score)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Has Diabetes'],
            yticklabels=['No Diabetes', 'Has Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Diabetes Prediction (SDG 3)')
# Important: Use plt.savefig('confusion_matrix.png') to save the image for the README.md
plt.show()


# ------------------------------------------------------------------------------
# 5. Ethical Reflection (Assignment Requirement)
# ------------------------------------------------------------------------------

print("\n--- Ethical & Social Reflection (Required for Submission) ---")

# SDG 3 Promotion: This model aids early detection in low-resource settings,
# contributing to Goal 3.4 (reducing premature mortality from NCDs) by identifying
# high-risk individuals for timely intervention.

# Reflection on Bias and Fairness:
# The Pima Indians dataset is known to be geographically and ethnically limited.
# Ethical Problem: If deployed widely, the model may exhibit **data bias** and
# underperform on diverse populations, potentially leading to incorrect diagnoses
# and exacerbating **healthcare inequality**.
# Mitigation: To ensure **fairness**, future development must involve diverse,
# multi-ethnic data collection and separate monitoring of metrics (e.g., Recall/Precision)
# across different demographic subgroups to ensure equitable performance.

print("Ethical reflection integrated into the code's documentation.")
# ------------------------------------------------------------------------------