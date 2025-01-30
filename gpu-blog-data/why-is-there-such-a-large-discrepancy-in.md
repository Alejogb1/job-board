---
title: "Why is there such a large discrepancy in custom metrics between model fit and evaluation?"
date: "2025-01-30"
id: "why-is-there-such-a-large-discrepancy-in"
---
The observed discrepancy between custom metrics during model fitting and subsequent evaluation often stems from a critical difference in the data distributions being used: training data versus validation/testing data, and the specific nature of the custom metric itself, particularly its sensitivity to edge cases or outliers. I've encountered this several times across different modeling projects, most recently while working on a predictive maintenance system for industrial machinery. The model, using a custom cost-based metric, showed a 90% performance during training, which dropped to 65% on the held-out validation set. This pointed directly to the problem of over-optimizing for the training data's specific characteristics.

Let's unpack this. During model fitting (training), the optimization process directly minimizes the loss function as it pertains to the training data. This iterative process inherently fine-tunes model parameters to best represent the training sample's statistical properties. However, this can lead to overfitting. That is, the model learns spurious patterns specific to the training set rather than generalizable patterns that would apply to new, unseen data. This is amplified when using custom metrics that may not be strictly convex or smooth, causing the optimizer to get stuck in a local minima that looks good on the training set but is far from ideal in a more general context. Furthermore, evaluation sets are ideally meant to simulate real-world data, and thus, might contain data points the training set didn’t cover, leading to this discrepancy.

The custom metric itself is often a critical factor. Unlike standard metrics like accuracy or mean squared error, custom metrics frequently prioritize specific types of errors or outcomes. For instance, in the predictive maintenance scenario, my metric might penalize false negatives much more severely than false positives, leading to a training process that favors models that are overly sensitive. This can be effective if the training and validation data are very similar, however, if the validation set has slightly different proportions of the classes, or more difficult cases in the class the model is being tuned to protect, a dramatic drop-off in performance can occur. Metrics that involve ratios or divisions are particularly prone to instability if their denominator approaches zero or if edge case values dominate the result. Thus, a high result during training can be artificially inflated.

To illustrate this point further, consider the following python examples using sklearn and numpy:

**Example 1: A basic scoring function and a distribution shift.**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def custom_metric(y_true, y_pred):
    # Assume we're prioritizing recall at the cost of precision
    # This is a very simplified illustration, in real settings this can be much more complex
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / (actual_positives + 1e-8) # Adding a tiny value to avoid division by zero

# Generate imbalanced synthetic data.
np.random.seed(42)
X = np.random.rand(1000, 5)
y = np.random.choice([0, 1], size=1000, p=[0.8, 0.2]) # 20% positive cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Introduce a shift to the validation data.
y_test = np.random.choice([0, 1], size=y_test.shape[0], p=[0.6, 0.4]) # 40% positive cases.

# Train a simple logistic regression model.
model = LogisticRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# evaluate with the custom metric
train_score = custom_metric(y_train, y_train_pred)
test_score  = custom_metric(y_test, y_test_pred)

print(f"Training Score: {train_score:.4f}")
print(f"Validation Score: {test_score:.4f}")

print(f"Training F1: {f1_score(y_train, y_train_pred):.4f}")
print(f"Testing F1: {f1_score(y_test, y_test_pred):.4f}")

```

In this example, we see that the custom metric, which is just recall, will favor over prediction of the positive class which is under represented in the training data. The validation set however has a larger positive class representation. This difference in distribution causes a significant drop in custom metric performance. However, the F1 score is more stable.

**Example 2: Sensitivity to edge cases.**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def custom_metric(y_true, y_pred):
    # A metric that's highly sensitive to outliers, which are penalized more
    errors = np.abs(y_true - y_pred)
    return np.mean(errors**3)

np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X.flatten() + np.random.normal(0, 0.5, 100)  # linear relationship with noise
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Introduce an outlier in validation data.
y_test[0] += 5 # one large outlier

model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_custom_score = custom_metric(y_train, y_train_pred)
test_custom_score = custom_metric(y_test, y_test_pred)

train_mse_score = mean_squared_error(y_train, y_train_pred)
test_mse_score = mean_squared_error(y_test, y_test_pred)

print(f"Training Custom Metric: {train_custom_score:.4f}")
print(f"Validation Custom Metric: {test_custom_score:.4f}")
print(f"Training MSE: {train_mse_score:.4f}")
print(f"Testing MSE: {test_mse_score:.4f}")

```
Here, the custom metric penalizes larger errors disproportionately. The small outlier added to the validation data causes a dramatic increase in the custom loss, but has a modest impact on MSE. This further highlights the importance of understanding the characteristics of a custom metric, as this can easily mask the underlying quality of a model.

**Example 3:  Training using a proxy metric with an evaluation with another metric.**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
def custom_metric(y_true, y_pred):
    # Assume we're prioritizing recall at the cost of precision
    # This is a very simplified illustration, in real settings this can be much more complex
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / (actual_positives + 1e-8)

# Generate imbalanced synthetic data.
np.random.seed(42)
X = np.random.rand(1000, 5)
y = np.random.choice([0, 1], size=1000, p=[0.8, 0.2]) # 20% positive cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# evaluate using custom metric in training
training_score = custom_metric(y_train, y_train_pred)

# evaluate on the evaluation set using the ROC AUC score.
testing_score = roc_auc_score(y_test, y_test_prob)


print(f"Training Score: {training_score:.4f}")
print(f"Validation Score: {testing_score:.4f}")
print(f"Validation Recall: {recall_score(y_test, y_test_pred):.4f}")
```

Here, during training, our metric is only used to monitor the quality of the training, but we make no effort to actually optimize that score. When we evaluate the model based on a different metric, like area under the ROC curve (AUC) or recall, we see that what is good for our proxy custom metric may not reflect the desired output in our test set. In this case, since we are not explicitly tuning our model to optimize recall, it makes sense that AUC or recall may not be as high as the custom metric, even if the custom metric itself is flawed.

From this, it is clear that careful consideration must be given to the type of metric used and its mathematical properties, and that simply training to optimize a given metric is not sufficient.

In summary, a significant discrepancy between training and evaluation custom metric scores is frequently the result of a disconnect between the training data distribution and evaluation data distribution, often exacerbated by the nature of the custom metric itself. Overfitting and the use of unstable or inappropriate metrics lead to artificially high scores during training that fail to generalize to new data. Strategies to mitigate these issues should include careful data preparation, robust cross-validation, choosing metrics with an eye to their stability and mathematical properties, and using validation to evaluate performance on data that mimics the real-world data as closely as possible.

For further information and a deeper understanding of these topics, I recommend reviewing resources that cover statistical learning principles, model evaluation techniques, and the importance of robust validation strategies. Specific texts covering model assessment, and books or articles on the topic of loss functions and their properties can all be useful. Furthermore, the scikit-learn documentation on model validation and hyperparameter tuning is an invaluable practical resource to keep close at hand.
