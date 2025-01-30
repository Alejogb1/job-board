---
title: "Why isn't all provided data used in model fitting?"
date: "2025-01-30"
id: "why-isnt-all-provided-data-used-in-model"
---
The underutilization of provided data in model fitting stems fundamentally from the inherent trade-off between model complexity and generalization performance.  My experience debugging and optimizing machine learning pipelines over the last decade has consistently highlighted this tension.  Simply including all available data points isn't always optimal; indeed, it can actively degrade the model's ability to predict unseen data.  This effect is particularly pronounced in scenarios involving high dimensionality, noisy data, or the presence of outliers.

**1.  The Explanation: Bias-Variance Trade-off and Data Quality**

The core issue lies in the bias-variance trade-off.  A model trained on the entirety of a dataset, especially one containing noise or irrelevant features, will likely exhibit low bias but high variance. Low bias implies the model is capable of fitting the training data well; it captures the underlying patterns effectively. However, high variance indicates that the model is overly sensitive to the specificities of the training set, including its noise and outliers. This overfitting leads to poor generalization—the model performs exceptionally well on the training data but poorly on new, unseen data.  Conversely, a simpler model, trained on a carefully selected subset of the data, may have higher bias (it doesn't perfectly capture all training data nuances), but significantly lower variance, resulting in better generalization.

Furthermore, the quality of the data itself plays a critical role.  Data might contain errors (incorrect labels, missing values, measurement inaccuracies), inconsistencies (duplicate entries, conflicting information), or irrelevant features (attributes that don't contribute to the predictive power of the model). Including such flawed data in model training can exacerbate overfitting and lead to unreliable predictions.  My work on a large-scale fraud detection project underscored this issue; including all reported transactions, without rigorous data cleaning and validation, led to a model that was highly accurate on the training data but performed poorly in real-world deployment due to a high proportion of noisy transactions in the original dataset.


**2. Code Examples and Commentary**

The following examples illustrate data selection strategies to improve model performance.  These are simplified illustrations and would require adaptation based on specific dataset characteristics and modeling goals.

**Example 1: Outlier Removal using IQR**

This example demonstrates outlier removal using the Interquartile Range (IQR) method. Outliers are frequently a source of noise that significantly impacts model fitting.

```python
import numpy as np
import pandas as pd

data = pd.DataFrame({'feature': np.random.normal(loc=0, scale=1, size=100)})
data.loc[np.random.choice(range(100), 10), 'feature'] = 10  # Introduce outliers

Q1 = data['feature'].quantile(0.25)
Q3 = data['feature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_data = data[(data['feature'] >= lower_bound) & (data['feature'] <= upper_bound)]

#Use filtered_data for model fitting
print(f"Original data size: {len(data)}")
print(f"Filtered data size: {len(filtered_data)}")
```

This code first generates a dataset with some artificially added outliers. It then calculates the IQR and uses it to define upper and lower bounds. Data points falling outside these bounds are deemed outliers and removed.  The resulting `filtered_data` is then used for model training, leading to a more robust model less sensitive to these extreme values.

**Example 2: Feature Selection using Recursive Feature Elimination (RFE)**

This example employs Recursive Feature Elimination (RFE), a technique that iteratively removes features based on their importance to the model. This addresses the issue of irrelevant features that can contribute to overfitting.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5) #Selecting top 5 features
X_selected = rfe.fit_transform(X, y)

# Train the model on the selected features
#...Model Training using X_selected and y
print(f"Original number of features: {X.shape[1]}")
print(f"Number of selected features: {X_selected.shape[1]}")

```

This code generates a synthetic dataset with many features, some irrelevant. RFE is applied to select the five most important features based on the logistic regression model. Only these features are then used for model training, enhancing efficiency and reducing overfitting due to irrelevant features.  The choice of `n_features_to_select` is crucial and often determined through cross-validation.


**Example 3: Data Subsampling using k-Fold Cross-Validation**

This example demonstrates k-fold cross-validation, a technique that partitions the data into k subsets. The model is trained on k-1 subsets and validated on the remaining subset. This approach mitigates overfitting by exposing the model to different parts of the dataset during training.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.random.rand(100, 5) # 100 samples, 5 features
y = 2*X[:,0] + 3*X[:,1] + np.random.randn(100) # Linear relationship with noise

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores)}")
```

This code uses k-fold cross-validation with five folds.  The model's performance is evaluated on each fold, providing a more robust estimate of its generalization ability than training and evaluating on the entire dataset.  The average score provides a better indication of the model's true performance on unseen data.

**3. Resource Recommendations**

For deeper understanding, I recommend consulting texts on statistical learning theory, focusing on topics like bias-variance trade-off, regularization techniques (L1, L2), and model selection methods (cross-validation, information criteria).  Furthermore,  texts detailing data preprocessing and feature engineering are invaluable, particularly those focusing on outlier detection, missing data imputation, and dimensionality reduction.  Finally, practical experience through working with real-world datasets and engaging with online communities devoted to machine learning will further solidify your understanding.
