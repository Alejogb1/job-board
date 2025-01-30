---
title: "Why is the model performing poorly on training and validation sets despite good performance on the test set?"
date: "2025-01-30"
id: "why-is-the-model-performing-poorly-on-training"
---
The phenomenon of a machine learning model exhibiting superior performance on a held-out test set while underperforming on both training and validation sets strongly suggests a data leakage issue, specifically one manifesting during the data splitting process.  My experience troubleshooting similar problems across numerous projects, from fraud detection systems to natural language processing tasks, points to this as the primary suspect.  This isn't a simple overfitting case; consistently poor performance across training and validation implies a fundamental flaw in how the data is being prepared and partitioned, not just an issue of model complexity.


**1. Clear Explanation:**

The crux of the problem lies in the relationship between the training, validation, and test sets.  These sets are intended to be independent and identically distributed (i.i.d.).  However, if information from the test set—either explicitly or implicitly—influences the training or validation processes, the model will appear to perform well on the test set due to an artificial advantage, while its actual generalization ability remains poor.  This is reflected in the poor training and validation scores.  This leakage can occur in various subtle ways:

* **Target Leakage:**  This is a common scenario. Information inadvertently included in the training data that is strongly correlated with the target variable, but would not be available during real-world prediction, leads to inflated performance on the test set.  For example, if predicting customer churn, including a feature representing a customer's future cancellation date in the training data (even accidentally) would lead to exceptionally high accuracy on the test set, but disastrous performance in production.

* **Feature Leakage:** Similar to target leakage, this occurs when features used for training contain information that is not available during deployment.  For instance, using information about future events or aggregated statistics computed across the entire dataset (including the test set) in feature engineering will artificially boost performance on the test set.  Time series data is particularly prone to this.

* **Data Splitting Issues:** A less obvious source of leakage stems from flawed data splitting procedures.  If the random seed used for splitting is not consistently set, or if stratification (ensuring proportional class representation in each set) is not properly implemented, the sets may not be truly independent, leading to information bleed-through.

* **Preprocessing Leakage:** Applying transformations, scaling, or imputation based on the entire dataset before splitting is another common pitfall.  The model effectively "sees" information from the test set during preprocessing, rendering the test set evaluation invalid.

Addressing these issues requires meticulous examination of the entire data pipeline, from raw data ingestion to model evaluation.


**2. Code Examples with Commentary:**

The following examples demonstrate how data leakage can subtly manifest and how to mitigate it.  I have simplified these for clarity; in real-world scenarios, the leakage can be far more insidious.

**Example 1: Target Leakage (Python with scikit-learn)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Simulate data with target leakage (a simplified example)
data = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200), 'target': range(100)})
data['leakage_feature'] = data['target'] + 1  # Leakage!

X = data[['feature1', 'feature2', 'leakage_feature']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Training Score: {model.score(X_train, y_train)}")
print(f"Test Score: {model.score(X_test, y_test)}")
```

In this example, `leakage_feature` is directly derived from the target.  The high test score is spurious.  Correcting this requires removing `leakage_feature`.


**Example 2: Feature Leakage (Python with pandas and scikit-learn)**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Simulate time series data with feature leakage
dates = pd.date_range('2023-01-01', periods=100)
data = pd.DataFrame({'date': dates, 'value': np.random.rand(100)})
data['future_value'] = data['value'].shift(-1) # Leakage!

# Incorrect splitting -  leakage occurs due to inappropriate feature creation 
X = data[['value','future_value']]
y = data['value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

print(f"Training Score: {model.score(X_train, y_train)}")
print(f"Test Score: {model.score(X_test, y_test)}")
```

Here, `future_value` contains information from the future.  The model achieves artificially high scores.  The solution: remove `future_value` or restructure feature engineering to avoid using future data.


**Example 3:  Preprocessing Leakage (Python with scikit-learn)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Simulate data
data = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})

# Incorrect preprocessing: scaling the entire dataset before splitting
scaler = StandardScaler()
data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])

X = data[['feature1', 'feature2']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Training Score: {model.score(X_train, y_train)}")
print(f"Test Score: {model.score(X_test, y_test)}")

# Correct preprocessing: scale each set independently
scaler_train = StandardScaler()
scaler_test = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)
X_test_scaled = scaler_test.transform(X_test)

model.fit(X_train_scaled, y_train)
print(f"Training Score (Correct): {model.score(X_train_scaled, y_train)}")
print(f"Test Score (Correct): {model.score(X_test_scaled, y_test)}")
```


This demonstrates the impact of scaling the entire dataset before splitting. The correct approach is to scale training and test sets independently to avoid leakage.



**3. Resource Recommendations:**

*  A comprehensive textbook on machine learning covering data preprocessing and model evaluation techniques in detail.
*  A practical guide to data wrangling and feature engineering in your chosen programming language.
*  Advanced statistical learning materials focusing on model diagnostics and assessing model assumptions.
* A reputable online course or workshop focusing specifically on the practical aspects of data preparation and model building.  Look for one emphasizing best practices for avoiding data leakage.
*  Documentation for your chosen machine learning libraries, paying close attention to functions related to data splitting and preprocessing.


By systematically investigating each stage of the data pipeline and applying the principles outlined above, you can effectively identify and resolve data leakage issues, resulting in a model that generalizes well and performs consistently across training, validation, and test sets.  Remember, rigorous data handling is paramount to building reliable and robust machine learning models.
