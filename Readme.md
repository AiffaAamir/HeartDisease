
# 🫀 Heart Disease Prediction with XGBoost + Hyperopt

This project predicts the likelihood of heart disease using machine learning techniques, particularly the XGBoost algorithm with advanced hyperparameter tuning via Hyperopt. The dataset is preprocessed, visualized, and evaluated to ensure a robust and reliable model — with impressive results (AUC > 0.98)!

## 📁 Project Overview

- **Dataset**: `heart_dataset.csv`
- **Task**: Binary classification (predicting heart disease presence)
- **Algorithm**: XGBoost Classifier
- **Tuning Method**: Hyperopt (Bayesian optimization)
- **Evaluation Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1-Score

---

## 📊 Dataset Exploration

```python
# Load data and view head
df = pd.read_csv("heart_dataset.csv")
print(df.head())
````

* Checked class distribution using bar chart.
* Counted and handled missing values.
* Encoded categorical data (`sex`) and filled numeric NaNs with median values.

---

## 🧹 Data Preprocessing

* Gender encoding: Male → 1, Female → 0
* Filled missing numeric values with median.
* Splitting features and labels.
* 80-20 train-test split.

---

## 🔍 Hyperparameter Tuning (Hyperopt)

We used **Bayesian optimization** with Hyperopt to tune the XGBoost hyperparameters such as:

* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`

```python
best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials
)
```

---

## 🧠 Model Training (XGBoost)

Used the optimized parameters and trained the model using XGBoost’s **low-level API** with `DMatrix` and **early stopping**.

```python
final_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dtest, "validation")],
    early_stopping_rounds=20
)
```

---

## 📈 Evaluation Results

| Metric    | Score |
| --------- | ----- |
| AUC-ROC   | 0.98+ |
| Accuracy  | \~96% |
| Precision | \~97% |
| Recall    | \~87% |
| F1-Score  | \~92% |

Confusion matrix and metric visualizations were plotted for deeper analysis.

---

## 📌 Key Takeaways

* Model is **highly accurate** and **reliable** for detecting heart disease.
* AUC and Precision indicate strong ranking and low false-positive rates.
* Balanced F1-score ensures both **precision** and **recall** are considered.

---

## 🚀 Future Improvements

* Add feature importance visualization.
* Use cross-validation for more robust tuning.
* Deploy the model as a web app using **Flask** or **Streamlit**.

---

## 📚 Requirements

```bash
pip install pandas numpy matplotlib scikit-learn xgboost hyperopt
```

---

## 📝 Author

**Aiffaa**
💡 Passionate about ML in healthcare and predictive analytics


