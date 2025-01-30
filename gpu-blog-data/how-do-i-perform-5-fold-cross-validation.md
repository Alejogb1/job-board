---
title: "How do I perform 5-fold cross-validation?"
date: "2025-01-30"
id: "how-do-i-perform-5-fold-cross-validation"
---
Implementing robust model evaluation through k-fold cross-validation, specifically with k=5, is a common practice I've employed extensively in my machine learning projects, particularly when working with datasets of moderate size. The core principle behind this approach involves partitioning the available data into 'k' equal-sized folds, and iteratively training and testing the model, ensuring each fold serves as both training and test data. This method mitigates overfitting by providing a more generalized performance estimate than a single train/test split. In the case of 5-fold cross-validation, the data is divided into five parts. Each part is used as a validation set once, and the other four parts are used as training set. I've found this to be particularly effective when dataset size doesn't permit for a large isolated validation dataset.

The process begins with shuffling the dataset to randomize the distribution of data points across folds. This crucial step prevents bias that could arise from any inherent order in the original dataset. Following this, the data is divided into five distinct folds. In each of the five iterations of cross-validation, a different fold serves as the validation set, while the remaining four folds are concatenated and used as the training set. This cycle ensures each data point is used exactly once for testing and four times for training. The model is trained on the training set, evaluated on the validation set, and a performance metric (e.g., accuracy, precision, recall, F1-score, RMSE, MAE) is calculated. These metrics are stored for each of the five iterations. Once the process is complete, the average and standard deviation of the evaluation metric across the five folds are calculated. These final values provide an estimated measure of the model's generalization performance, and the standard deviation provides insight into the stability of the performance.

The advantage of 5-fold, or generally k-fold, cross-validation, over a single train/test split is its robustness. Because each sample has a chance to be part of the testing set, we gain more comprehensive insight into the model’s performance on unseen data. Single train/test splits are susceptible to overestimating model performance depending on the specific random split. With k-fold cross-validation, we average the scores across multiple iterations, which results in a more reliable estimate of the model’s real-world performance, particularly in the presence of noisy data or small dataset sizes. I've seen the difference firsthand, where models that appeared strong on a single split performed much worse on cross-validation.

Here are some examples, illustrating practical implementation, in Python using scikit-learn, a library I frequently use for machine learning tasks.

**Example 1: Basic Implementation using `KFold` and `cross_val_score`**

This first example demonstrates the essential usage of `KFold` to create the folds and `cross_val_score` to automatically run the cross-validation and provide performance scores. I often prefer this approach when dealing with models where the evaluation pipeline is straightforward.

```python
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get accuracy scores
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Print the average accuracy and standard deviation
print(f"Accuracy Scores: {scores}")
print(f"Average Accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation: {np.std(scores):.4f}")
```

This code snippet first generates synthetic data for demonstration.  I use the `LogisticRegression` model for illustration, but the procedure works with any other scikit-learn model. The `KFold` class with `n_splits=5` creates the cross-validation folds, ensuring the data is shuffled prior to creating the splits for a fair split using the parameter `shuffle=True`. The `cross_val_score` function does all of the heavy lifting, automatically training and evaluating the model for each fold using 'accuracy' as the evaluation metric and using the `KFold` instance as splitting strategy. Finally, I print the array of scores, the average accuracy, and the standard deviation across folds. The 'liblinear' solver was explicitly specified to handle this binary classification problem.

**Example 2: Manual Implementation with `KFold`**

In cases where I require finer control over the training process, such as implementing a custom pre-processing step or a custom evaluation loop, I use `KFold` to manually control the train and test splits.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store accuracy scores
accuracy_scores = []

# Iterate through folds
for train_index, test_index in kf.split(X):
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Append to the scores
    accuracy_scores.append(accuracy)

# Print the average accuracy and standard deviation
print(f"Accuracy Scores: {accuracy_scores}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation: {np.std(accuracy_scores):.4f}")
```

This example replicates the previous one but manually iterates over the `KFold` splits. Each iteration provides the indices for train and test sets. Inside the loop, I extract the training and test data, train the model using the training data, perform predictions using the test set, and finally calculate accuracy. This is a more hands-on method which is useful to integrate custom pre-processing steps such as scaling, feature engineering or evaluation metrics which cannot be passed to `cross_val_score`. Finally, the average accuracy and standard deviation are calculated and displayed.

**Example 3: Cross-validation with Stratification for Imbalanced Datasets**

When working with datasets that have imbalanced class distributions, standard k-fold cross-validation can lead to issues where one or more folds may not adequately represent the minority class. In such scenarios, I've found `StratifiedKFold` beneficial, as it preserves the class proportions across the folds.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Generate imbalanced synthetic data
X, y = make_classification(n_samples=100, n_features=20, weights=[0.9, 0.1], random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)

# Initialize StratifiedKFold with 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store accuracy scores
accuracy_scores = []

# Iterate through folds
for train_index, test_index in skf.split(X, y):
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Append to the scores
    accuracy_scores.append(accuracy)


# Print the average accuracy and standard deviation
print(f"Accuracy Scores: {accuracy_scores}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation: {np.std(accuracy_scores):.4f}")

```
This example uses `make_classification` to create an imbalanced dataset and then uses `StratifiedKFold` to generate splits. The main change here is passing `y` as the second argument to `skf.split`. This ensures that the class ratio in training and test sets are similar. The rest of the code follows the manual implementation pattern as before. This stratification can greatly improve your performance analysis with imbalanced datasets.

In summary, 5-fold cross-validation is a valuable technique for reliably evaluating model performance. It's important to be aware of the assumptions and limitations of different evaluation metrics and the need to use stratified cross-validation when dealing with imbalanced data. I have found that the methods outlined in these examples have consistently provided robust performance estimates in my projects. As for further study, exploring resources that cover model evaluation techniques beyond simple cross-validation, such as nested cross-validation, is highly recommended, and in-depth studies of scikit-learn’s documentation. Additionally, texts on statistical learning theory can offer a deeper understanding of cross-validation and its theoretical underpinnings. These resources provide a foundation for choosing appropriate model evaluation strategies for various machine learning problems.
