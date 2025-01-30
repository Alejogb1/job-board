---
title: "How to split a dataset into training, validation, and test sets?"
date: "2025-01-30"
id: "how-to-split-a-dataset-into-training-validation"
---
Dataset partitioning for model training, validation, and testing is crucial for robust machine learning.  My experience in developing high-performance recommendation systems consistently highlights the importance of stratified sampling to maintain class distribution across the partitions.  Failing to account for class imbalance during this stage can lead to biased model performance estimates and ultimately, a suboptimal deployed model.

**1. Clear Explanation:**

The process of splitting a dataset involves dividing it into three mutually exclusive subsets: the training set, the validation set, and the test set.  Each serves a distinct purpose in the machine learning workflow.

* **Training Set:** This is the largest portion of the data and is used to train the machine learning model. The model learns patterns and relationships from this data to make predictions. The algorithm adjusts its internal parameters based on the training data to minimize error.

* **Validation Set:** This subset is used to tune hyperparameters and evaluate the model's performance during training.  It provides an unbiased estimate of the model's generalization ability on unseen data *during* the model development process. This allows for adjustments to the model architecture or training process without being influenced by the final test set.  Overfitting to the validation set is a concern, thus it's crucial to carefully monitor this process.

* **Test Set:** This is held out entirely until the model is finalized.  It's used for a single, final evaluation of the model's performance on completely unseen data. This provides the most realistic estimate of how the model will perform in a real-world deployment scenario.  The test set should never be used during model development or hyperparameter tuning.

The optimal proportions for these subsets are highly dependent on the size of the dataset and the complexity of the problem. However, common ratios include 70% for training, 15% for validation, and 15% for testing.  For smaller datasets, a larger validation set might be beneficial to get more reliable performance estimates during hyperparameter tuning.  Stratified sampling, ensuring representative class proportions in each subset, is always recommended, particularly with imbalanced datasets.

**2. Code Examples with Commentary:**

These examples demonstrate dataset splitting using Python and popular libraries.  They incorporate stratified sampling for robust performance evaluation.

**Example 1: Using scikit-learn's `train_test_split` (Simple Split):**

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 2, 100)  # Binary classification labels

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets (75/25 split of the training data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

This example uses `train_test_split` twice. First, it splits the data into training and testing sets.  Then it further splits the training set into training and validation sets.  `random_state` ensures reproducibility.  This method lacks stratification.


**Example 2: Using scikit-learn's `StratifiedShuffleSplit` (Stratified Split):**

```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Sample data (replace with your actual data)
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train_test, X_test = X[train_index], X[test_index]
    y_train_test, y_test = y[train_index], y[test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, val_index in sss.split(X_train_test, y_train_test):
    X_train, X_val = X_train_test[train_index], X_train_test[val_index]
    y_train, y_val = y_train_test[train_index], y_train_test[val_index]

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

```

This example uses `StratifiedShuffleSplit` which preserves the percentage of samples for each class in each split, addressing class imbalance. This is achieved by splitting the data in a way that maintains the proportion of classes in each subset.  It’s a more robust approach compared to the previous example.

**Example 3: Manual Stratified Splitting (for deeper understanding):**

```python
import pandas as pd
import numpy as np

# Sample data (replace with your actual data, assuming 'target' column represents the labels)
data = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})

# Calculate class proportions
class_proportions = data['target'].value_counts(normalize=True)

# Create stratified subsets
train_df = data.groupby('target').sample(frac=0.7, random_state=42)
remaining_df = data.drop(train_df.index)
val_df = remaining_df.groupby('target').sample(frac=0.5, random_state=42)
test_df = remaining_df.drop(val_df.index)

# Verify class proportions (optional)
print("Training set proportions:\n", train_df['target'].value_counts(normalize=True))
print("Validation set proportions:\n", val_df['target'].value_counts(normalize=True))
print("Test set proportions:\n", test_df['target'].value_counts(normalize=True))

# Separate features and target variables.
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_val = val_df.drop('target', axis=1)
y_val = val_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

```

This example demonstrates a manual stratified split using pandas. It groups the data by the target variable and then samples a fraction of each class for each subset. This provides a more granular control over the stratification process, offering deeper insight into the dataset's structure.


**3. Resource Recommendations:**

For a deeper understanding of dataset partitioning techniques, I recommend consulting reputable textbooks on machine learning and statistical learning.  In addition, review the documentation for popular machine learning libraries such as scikit-learn and explore online tutorials focusing on best practices for data preprocessing and model evaluation.  Consider exploring more advanced techniques like k-fold cross-validation for improved model evaluation, particularly with limited datasets.  Finally, dedicated publications on imbalanced learning can provide crucial insight when dealing with datasets where class representation is skewed.
