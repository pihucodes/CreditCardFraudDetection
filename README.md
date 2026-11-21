# Credit Card Fraud Detection

Machine-learning project that detects fraudulent credit card transactions using a **Random Forest** classifier, with a **Logistic Regression** baseline and an interactive **Streamlit** dashboard.

## Problem
Fraud is rare, so it is hard to detect. In this dataset only **492 of 284,807 transactions (0.17%)** are fraud. A model that always predicts "legitimate" would score 99.83% accuracy while catching nothing, so this project is judged on **precision, recall, F1 and PR/ROC-AUC**, not accuracy.

## Dataset
- Source: [Credit Card Fraud Detection (ULB) on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions by European cardholders (September 2013), 31 columns
- `Time`, `Amount`, `V1`-`V28` (anonymised PCA components), `Class` (0 = legitimate, 1 = fraud)
- No missing values
- The file is included as `creditcard.zip`. Unzip it to get `creditcard.csv` in the project folder.

## Method
1. Load and inspect the data, check class imbalance
2. Stratified 80/20 train-test split (`random_state=42`)
3. **Random Forest**: 100 trees, `max_depth=12`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`
4. **Logistic Regression** baseline with `StandardScaler` and balanced class weights
5. Evaluate with precision, recall, F1, confusion matrix, PR curve, ROC curve
6. Threshold analysis (0.1 to 0.9)
7. Save the model with `joblib` and serve it through Streamlit

## Results (test set: 56,962 transactions, 98 frauds)

| Model | Accuracy | Precision | Recall | F1 | Avg Precision | ROC AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9755 | 0.0610 | 0.9184 | 0.1144 | 0.7190 | 0.9721 |
| Random Forest | 0.9994 | 0.8421 | 0.8163 | 0.8290 | 0.8466 | 0.9809 |

Random Forest at threshold 0.5 caught 80 of 98 frauds with 15 false alarms. Most important features: V14, V10, V12, V17, V4.

## Project structure
```
app.py                                  Streamlit dashboard
credit_card.ipynb                       Full notebook (training and evaluation)
Credit_Card_Fraud_Detection-Copy1.html  Notebook exported with outputs
fraud_dashboard.html                    Static results summary page
credit_fraud_random_forest.pkl          Trained model
model_features.pkl                      Feature names the model expects
creditcard.zip                          Dataset
requirements.txt                        Python dependencies
```

## How to run
```bash
pip install -r requirements.txt
# unzip creditcard.zip so creditcard.csv is in this folder
streamlit run app.py
```
To retrain the model, open `credit_card.ipynb` in Jupyter and run all cells.

## Dashboard features
- Upload a CSV of transactions or use a demo sample
- Adjustable fraud threshold
- Flagged transactions ranked by risk, with CSV download
- Probability histogram and feature importance chart
- Live precision, recall, F1 and confusion matrix when a `Class` column is present
- Handles missing or non-numeric values

## Limitations and future work
- Single train/test split, no cross-validation
- Threshold chosen on the test set; a separate validation set would be better
- Compare with XGBoost, SMOTE and cost-sensitive learning
- Features are anonymised, so results can't be explained in business terms

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit, Jupyter

## Authors
<add your names here>
