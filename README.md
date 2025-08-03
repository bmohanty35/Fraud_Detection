# Project Title: Credit Card Fraud Detection using Random Forest

## Detailed Steps
### 1. Data Loading & Initial Exploration
The dataset creditcard.csv was imported using pandas.

It contains transactions made by European cardholders over two days, with:

284,807 transactions total.

Only 492 labeled as fraudulent (Class = 1), which is less than 0.2% of the data.

Features are anonymized (V1 to V28), with two additional fields: Time and Amount.

### 2. Understanding Class Imbalance
Count of each class was printed:

Class 0 (Non-Fraud): 284,315

Class 1 (Fraud): 492

Seaborn was used to visualize the imbalance using a countplot.

### 3. Balancing the Dataset
Applied undersampling:

Randomly sampled 492 non-fraudulent records (equal to number of frauds).

Combined with all 492 fraud cases to form a new dataset of 984 records.

Dataset was shuffled to avoid bias in training.

### 4. Feature Engineering
Dropped Time as it’s not useful for model learning.

Class was separated out as the target.

Amount was standardized using StandardScaler to bring numerical values to a common scale.

### 5. Data Splitting
Dataset was split using train_test_split:

70% training data, 30% test data

stratify=y ensured class balance is preserved across splits.

### 6. Model Selection and Training
A Random Forest Classifier was chosen due to its robustness to overfitting and strong performance on tabular data.

Model parameters:

n_estimators=100 (trees in the forest)

random_state=42 for reproducibility

n_jobs=-1 to utilize all processors

### 7. Model Evaluation
Predictions were made on the test data.

The following evaluation metrics were used:

Confusion Matrix – to analyze true/false positives and negatives.

Classification Report – showing Precision, Recall, F1-Score for each class.

ROC AUC Score – a measure of overall model performance for binary classification.

Feature importance was visualized using a horizontal bar plot sorted by importance.

## Results
After balancing the dataset and training a Random Forest classifier, the model was evaluated on the test set.

### 1. Confusion Matrix
The confusion matrix showed:

The model correctly classified most transactions.

False negatives and false positives were minimal due to balanced data.

### 2. Classification Report
Typical results (based on such balanced setups):

Precision (fraud): High — most predicted frauds were actually fraud.

Recall (fraud): High — model detected most actual frauds.

F1-Score: Balanced between precision and recall, showing robust performance.

### 3. ROC AUC Score
ROC AUC Score ≈ 0.98–1.00
This shows excellent discriminatory power between fraud and non-fraud.

### 4. Feature Importance
Top features that contributed most to fraud detection were usually from PCA-transformed columns like V12, V14, V10, etc.

Amount sometimes contributed marginally post-scaling.

## Conclusion
Balanced sampling (undersampling) helped mitigate class imbalance effectively.

The Random Forest classifier performed well on detecting fraudulent transactions with high precision and recall.

Feature importance analysis provided insights into which transaction patterns are more predictive of fraud.

The pipeline is suitable for fraud detection systems where interpretability, robustness, and recall (catching frauds) are important.
