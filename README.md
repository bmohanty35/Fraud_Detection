# Fraud_Detection

## Credit Card Fraud Detection using Random Forest
A machine learning project to detect fraudulent credit card transactions using a balanced dataset and a Random Forest classifier. The project addresses severe class imbalance and evaluates the model using various performance metrics.

## Project Overview
Credit card fraud detection is a classic imbalanced classification problem where fraud cases (positive class) are extremely rare.

This project uses undersampling to balance the dataset and Random Forest to classify transactions.

Includes detailed EDA, preprocessing, training, and evaluation stages.

## Dataset
File used: creditcard.csv

Features: anonymized PCA components (V1 to V28), Amount, Time

Target: Class (0 = Non-Fraud, 1 = Fraud)

## Workflow
### 1. Load and Inspect Dataset
Read the dataset with pandas.

Print and visualize the original class distribution.

Plot count of fraud vs non-fraud transactions using seaborn.

### 2. Class Imbalance Handling
Separate fraud and non-fraud records.

Use undersampling: randomly sample non-fraud transactions to match the number of fraud cases.

Combine and shuffle to form a balanced dataset.

Re-plot the class distribution to confirm balance.

### 3. Feature Engineering
Drop Time column.

Standardize the Amount feature using StandardScaler.

### 4. Split Data
Use train_test_split with stratify=y to maintain class balance in training and test sets.

Split into 70% train, 30% test.

### 5. Train Model
Use RandomForestClassifier from scikit-learn.

Set n_estimators=100 and enable parallel training with n_jobs=-1.

Fit the model on the training data.

### 6. Evaluate Model
Predict on test set.

Print confusion matrix and classification report (precision, recall, F1-score).

Calculate ROC AUC score using predicted probabilities.

### 7. Feature Importance
Extract feature importance from the trained Random Forest model.

Visualize top contributing features using a horizontal bar chart (Seaborn).
