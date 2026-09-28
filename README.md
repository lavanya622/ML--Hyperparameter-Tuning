# Hyperparameter Tuning with RandomizedSearchCV

## Overview

Hyperparameter tuning is the process of finding suitable hyperparameter values for a machine learning model to improve its performance.

In this practical, **RandomizedSearchCV** was used to tune a **Random Forest model**. Since the dataset contains time-ordered stock market data, **TimeSeriesSplit** was used for cross-validation instead of ordinary random cross-validation.

---

## Dataset

The dataset contains historical stock market information.

### Dataset Columns

| Column   | Description    |
| -------- | -------------- |
| `Date`   | Trading date   |
| `Open`   | Opening price  |
| `High`   | Highest price  |
| `Low`    | Lowest price   |
| `Close`  | Closing price  |
| `Volume` | Trading volume |

The dataset contains historical observations arranged in chronological order.

---

## Objective

The objectives of this practical are to:

* Understand hyperparameters in Random Forest
* Define a hyperparameter search space
* Use `RandomizedSearchCV`
* Apply time-series cross-validation
* Find the best hyperparameter combination
* Find the best cross-validation R² score
* Obtain the best Random Forest model

---

## Model Used

### Random Forest

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

The Random Forest model was used as the base estimator for hyperparameter tuning.

---

## Hyperparameters Tuned

The following Random Forest hyperparameters were included in the search:

```python
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}
```

### `n_estimators`

Controls the number of decision trees in the Random Forest.

### `max_depth`

Controls the maximum depth of each decision tree.

### `min_samples_split`

Defines the minimum number of samples required to split an internal node.

### `min_samples_leaf`

Defines the minimum number of samples required to be present in a leaf node.

### `max_features`

Controls the number of features considered when finding the best split.

---

## Time Series Cross-Validation

Because the dataset contains a `Date` column and represents time-ordered observations, **TimeSeriesSplit** was used.

```python
tscv = TimeSeriesSplit(n_splits=5)
```

TimeSeriesSplit maintains the chronological order of the observations during cross-validation.

This is useful for time-series data because future observations should not be used to train a model that is being evaluated on earlier observations.

---

## RandomizedSearchCV

`RandomizedSearchCV` searches through randomly selected combinations from the specified hyperparameter distributions.

The following settings were used:

```python
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)
```

### Important Parameters

* **Estimator:** Random Forest model
* **Search method:** Randomized Search
* **Number of combinations:** 20
* **Cross-validation:** TimeSeriesSplit with 5 splits
* **Scoring metric:** R²
* **Random state:** 42
* **n_jobs:** `-1` to use all available CPU cores

---

## Model Training

The RandomizedSearchCV object was fitted using the training data.

```python
random_search.fit(X_train_scaled, y_train)
```

The search evaluates different combinations of hyperparameters and identifies the combination with the best cross-validation R² score.

---

## Best Parameters

The best hyperparameter combination was obtained using:

```python
print("Best Parameters:")
print(random_search.best_params_)
```

The actual best parameters are available in the notebook output.

---

## Best Cross-Validation Score

The best cross-validation R² score was obtained using:

```python
print("Best CV R2 Score:")
print(random_search.best_score_)
```

The actual score is available in the notebook output.

---

## Best Model

The best-performing Random Forest estimator was extracted using:

```python
best_rf_model = random_search.best_estimator_
```

This model contains the hyperparameter combination selected by `RandomizedSearchCV`.

---

## Workflow

```text
Load Dataset
      ↓
Inspect Data
      ↓
Data Preprocessing
      ↓
Prepare Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Create Random Forest Model
      ↓
Define Hyperparameter Search Space
      ↓
TimeSeriesSplit
      ↓
RandomizedSearchCV
      ↓
Train Search
      ↓
Find Best Parameters
      ↓
Find Best CV R² Score
      ↓
Extract Best Model
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook / VS Code

---

## Machine Learning Concepts Learned

Through this practical, I learned:

* Hyperparameters
* Hyperparameter Tuning
* Random Forest
* RandomizedSearchCV
* Time Series Cross-Validation
* TimeSeriesSplit
* Cross-Validation
* R² Score
* Best Parameters
* Best Estimator
* Model Optimization

---

## Key Learning

This practical helped me understand how hyperparameters affect the performance of a machine learning model.

I learned how to define a search space for Random Forest and use `RandomizedSearchCV` to automatically search for suitable hyperparameter combinations.

I also learned why **TimeSeriesSplit** is appropriate when working with chronological data.

---

## Conclusion

In this practical, **RandomizedSearchCV** was applied to a Random Forest model using historical stock market data.

A set of Random Forest hyperparameters was defined, and 20 randomly selected combinations were evaluated using **5-fold TimeSeriesSplit cross-validation** with R² as the scoring metric.

The best hyperparameter combination and corresponding cross-validation score were obtained, and the best estimator was stored for further prediction and evaluation.

---
