# -*- coding: utf-8 -*-
"""HeartDisease Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v6ryKV_4rMQUyG_32SwlIg17rcQvFTcY
"""

# prompt: code to load data and show head

import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('heart_dataset.csv')

# Display the first 5 rows of the DataFrame
print(df.head())

# prompt: write a code of barchat to see how many volume each class have

import matplotlib.pyplot as plt

# Count the occurrences of each class
class_counts = df['target'].value_counts()
print(class_counts)

# Create the bar chart
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.title('Volume per Class')
plt.xlabel('Class')
plt.ylabel('Volume')
plt.xticks(rotation=0)  # Keep class names horizontal
plt.grid(axis='y', linestyle='--')
plt.show()

# prompt: write code to count missing values in each column

# Count the number of missing values in each column
missing_values_count = df.isnull().sum()

# Print the count of missing values for each column
print("\nMissing values count per column:")
missing_values_count

# Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 👇 Basic data preprocessing
def process_data(df):
    # Encode "sex" column: Male = 1, Female = 0
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    # Fill missing values with median (only for numeric columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df

# 📦 Apply preprocessing
df = process_data(df)

# 🔀 Split into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# ✂️ Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Confirm shape
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

"""Step-by-Step: Hyperparameter Tuning + Model Training"""



from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ⚙️ Define the objective function for hyperparameter tuning
def objective(params):
    # Ensure max_depth is integer
    params['max_depth'] = int(params['max_depth'])

    model = xgb.XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **params
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    return -auc  # We minimize loss

# 🔍 Search space for tuning
space = {
    'max_depth': hp.quniform('max_depth', 3, 6, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),  # log scale between 0.0067 and 1
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
    'reg_lambda': hp.uniform('reg_lambda', 1, 10),
    'min_child_weight': hp.uniform('min_child_weight', 0, 10)
}

# 🧪 Run Hyperopt
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials
)


print("✅ Best hyperparameters:", best_params)

# 🧠 Handle class imbalance
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# 🛠️ Fix Hyperopt parameter types
best_params['max_depth'] = int(best_params['max_depth'])

# 🧪 Convert data into DMatrix (XGBoost's internal format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 🧰 Create full params dictionary
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "scale_pos_weight": scale_pos_weight,
    **best_params
}

# 🏋️ Train with early stopping using low-level API
final_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dtest, "validation")],
    early_stopping_rounds=20,
    verbose_eval=True
)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# Prepare test data DMatrix
dtest = xgb.DMatrix(X_test)

# Predict probabilities and classes
y_proba = final_model.predict(dtest)  # probabilities
y_pred = (y_proba >= 0.5).astype(int)  # binary predictions with 0.5 threshold

# Print metrics
print("=== Model Evaluation Metrics ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

