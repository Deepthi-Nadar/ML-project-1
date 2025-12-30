# Credit Card Fraud Detection using Machine Learning

This project implements a machine learning–based system to detect fraudulent credit card transactions.  
It handles severe class imbalance using SMOTE and evaluates the model using a confusion matrix and ROC–AUC score. A simple real-time transaction detection simulation is also included.

---

## Project Description

Credit card fraud detection is a critical problem in the financial domain, where fraudulent transactions occur far less frequently than legitimate ones. Because of this imbalance, traditional evaluation metrics alone are not sufficient.  
This project focuses on building a reliable classification model that can effectively identify fraud while maintaining strong overall performance.

---

## Dataset Information

- Anonymized credit card transaction dataset
- Features:
  - `Time`
  - `V1` to `V28` (PCA-transformed features)
  - `Amount`
- Target variable:
  - `Class`
    - `0` → Legitimate transaction
    - `1` → Fraudulent transaction
- The dataset is highly imbalanced.

---

## Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Google Colab

---

## Methodology

1. Loaded and explored the dataset.
2. Verified data types and checked for missing values.
3. Split the dataset into training and testing sets using stratified sampling.
4. Applied SMOTE to balance the training dataset.
5. Trained a Logistic Regression model.
6. Evaluated the model using confusion matrix and ROC–AUC score.
7. Simulated real-time fraud detection using predicted probabilities.

---

## Model Evaluation

### Confusion Matrix
[[56213   651] [   10    88]]

### ROC–AUC Score
0.9759

The ROC–AUC score shows that the model has a strong ability to distinguish between fraudulent and legitimate transactions.

---

## Real-Time Detection Simulation

A basic real-time simulation checks transactions one by one using probability scores.

- Fraud is flagged when probability > **0.8**
- Otherwise, the transaction is considered legitimate

Sample output:
Legit Transaction. Probability: 0.00 🚨 FRAUD ALERT! Probability: 0.91


---

## Project Structure

CreditCardFraudDetection/
│
├── CreditCardFraudDetection.ipynb   # Containing data analysis, model training, and evaluation
├── creditcard.csv                  # Dataset used for training and testing the model
└── README.md                       # Project documentation

## Future Enhancements

- Test advanced models such as Random Forest and XGBoost
- Improve preprocessing and feature scaling
- Deploy the model as a web-based application
- Extend real-time detection to streaming data

---


