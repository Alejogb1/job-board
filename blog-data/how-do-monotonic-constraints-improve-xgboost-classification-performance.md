---
title: "How do monotonic constraints improve XGBoost classification performance?"
date: "2024-12-23"
id: "how-do-monotonic-constraints-improve-xgboost-classification-performance"
---

Alright, let's tackle this one. I've seen monotonic constraints in xgboost do some remarkable things, and they’re often underutilized. It's not just about magically improving numbers; it's about injecting domain knowledge into the model, making it more interpretable and sometimes even robust. I remember back when I was working on a project predicting customer churn for a telecom provider; we initially had an xgboost model that, while decent, produced some counterintuitive results, particularly concerning the relationship between monthly service cost and churn probability. That’s when we really started to appreciate the power of these constraints.

Monotonic constraints in xgboost, at their core, are all about ensuring that the model's predictions follow a predefined, consistent direction with respect to specific features. In simpler terms, if you specify that a feature is monotonically increasing, then as that feature's value increases, the model's predicted probability should also, at the very least, not decrease, all other factors remaining constant. Conversely, for monotonically decreasing constraints, the prediction should not increase as the feature increases. This isn't something achieved by accident or inherent to xgboost by default; we have to explicitly declare it.

Now, xgboost, without monotonic constraints, is designed to capture complex and sometimes non-linear relationships, and that's what makes it powerful. However, this flexibility can be a double-edged sword. When we have prior information about the expected direction of these relationships, letting xgboost figure it out can lead to overfitting, especially with limited data in specific feature ranges. We might end up with models that are difficult to trust precisely because they learn nuanced correlations that contradict domain understanding. Monotonicity imposes a restriction on the model that can lead to more consistent and, in certain contexts, improved performance by preventing overfitting in these crucial areas.

For example, consider credit risk assessment. A customer's income should ideally be monotonically increasing in relation to their credit score. A higher income should translate to a better or equal credit risk assessment, assuming everything else remains unchanged. Without constraints, an xgboost model might, by chance or through subtle overfitting, associate a higher income with a *slightly* worse credit score in some corner case of the dataset, and that would be nonsensical and difficult to debug in production. This inconsistency is where monotonic constraints step in.

The implementation in xgboost is relatively straightforward. You can specify the constraints during model training using the `monotone_constraints` parameter in the xgboost API. This parameter takes a tuple of integers, where `1` indicates an increasing constraint, `-1` indicates a decreasing constraint, and `0` indicates no constraint. Crucially, the order of these integers in the tuple must match the order of features passed to the xgboost model. This parameter is applied when splitting nodes during the tree building process, and the model is constrained to only choose splits that preserve the specified monotonic behavior.

Here's a code snippet illustrating this, assuming we're working with python:

```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
np.random.seed(42)
X = np.random.rand(100, 3) # Three features
y = (X[:,0] * 2 + X[:,1] * -1 + X[:,2] > 1).astype(int) # Target variable based on a function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training without monotonic constraints
dtrain_no_constraints = xgb.DMatrix(X_train, label=y_train)
dtest_no_constraints = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42}
model_no_constraints = xgb.train(params, dtrain_no_constraints, num_boost_round=10)
predictions_no_constraints = (model_no_constraints.predict(dtest_no_constraints) > 0.5).astype(int)
accuracy_no_constraints = accuracy_score(y_test, predictions_no_constraints)

print(f"Accuracy without constraints: {accuracy_no_constraints:.4f}")

# Training with monotonic constraints (Feature 0: Increasing, Feature 1: Decreasing, Feature 2: No constraint)
dtrain_with_constraints = xgb.DMatrix(X_train, label=y_train)
dtest_with_constraints = xgb.DMatrix(X_test, label=y_test)
params_with_constraints = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42, 'monotone_constraints': (1, -1, 0)}
model_with_constraints = xgb.train(params_with_constraints, dtrain_with_constraints, num_boost_round=10)
predictions_with_constraints = (model_with_constraints.predict(dtest_with_constraints) > 0.5).astype(int)
accuracy_with_constraints = accuracy_score(y_test, predictions_with_constraints)

print(f"Accuracy with constraints: {accuracy_with_constraints:.4f}")
```

In this snippet, we generated synthetic data with a known relationship between features and the target, and set up the constraints accordingly. In some runs you might see a minor accuracy boost and in others it might be similar to the unconstrained model depending on how lucky we are with splitting data and the relationship itself. However, the constraints lead to more *interpretable* models.

Let's consider another scenario. Suppose we have a dataset that examines the effect of hours of study on passing an exam. It is highly likely that more hours of study (within a realistic range) will lead to a higher probability of passing the exam. Let's model this with constraints:

```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Hours of study (0-10)
y = (0.2 * X[:, 0] + np.random.normal(0,0.5,100) > 0.5).astype(int)  # Target variable: pass or fail with noise


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training with monotonic constraints
dtrain_with_constraints = xgb.DMatrix(X_train, label=y_train)
dtest_with_constraints = xgb.DMatrix(X_test, label=y_test)
params_with_constraints = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42, 'monotone_constraints': (1,)}
model_with_constraints = xgb.train(params_with_constraints, dtrain_with_constraints, num_boost_round=10)
predictions_with_constraints = (model_with_constraints.predict(dtest_with_constraints) > 0.5).astype(int)
accuracy_with_constraints = accuracy_score(y_test, predictions_with_constraints)

print(f"Accuracy with constraint: {accuracy_with_constraints:.4f}")


#Training without constraints
dtrain_no_constraints = xgb.DMatrix(X_train, label=y_train)
dtest_no_constraints = xgb.DMatrix(X_test, label=y_test)

params_no_constraints = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42}
model_no_constraints = xgb.train(params_no_constraints, dtrain_no_constraints, num_boost_round=10)
predictions_no_constraints = (model_no_constraints.predict(dtest_no_constraints) > 0.5).astype(int)
accuracy_no_constraints = accuracy_score(y_test, predictions_no_constraints)

print(f"Accuracy without constraint: {accuracy_no_constraints:.4f}")
```
Here we can see that the accuracy is almost identical, however by visualizing the models using `xgboost.plot_importance` and `xgboost.plot_tree` we will clearly see that the constrained model will not display counterintuitive relationships for the constraint feature. This is very important for understanding and debugging our model.

Let's illustrate a final complex case. Suppose we are modeling customer satisfaction. There might be a feature capturing the number of calls made to support. With fewer calls we expect satisfaction to increase, and with higher calls we expect satisfaction to decrease until a certain threshold where additional calls do not influence satisfaction negatively anymore because the customer has clearly given up or found a workaround, but satisfaction won't increase either. This requires a piecewise monotonicity constraint, which we can approximate using multiple features to represent different regions of calls made to support:

```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
np.random.seed(42)
X_calls = np.random.rand(100, 1) * 10
X = np.concatenate([X_calls, np.maximum(0,X_calls - 5)], axis=1)  # Two features: calls and calls capped at 5 to help approximate the piecewise monotonic feature
y = (1 - 0.4*X_calls[:,0] + 0.1* np.maximum(0,X_calls[:,0]-5) + np.random.normal(0,0.3,100) > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training with monotonic constraints
dtrain_with_constraints = xgb.DMatrix(X_train, label=y_train)
dtest_with_constraints = xgb.DMatrix(X_test, label=y_test)
params_with_constraints = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42, 'monotone_constraints': (-1,1)}
model_with_constraints = xgb.train(params_with_constraints, dtrain_with_constraints, num_boost_round=10)
predictions_with_constraints = (model_with_constraints.predict(dtest_with_constraints) > 0.5).astype(int)
accuracy_with_constraints = accuracy_score(y_test, predictions_with_constraints)

print(f"Accuracy with piecewise constraint: {accuracy_with_constraints:.4f}")

# Training without constraints
dtrain_no_constraints = xgb.DMatrix(X_train, label=y_train)
dtest_no_constraints = xgb.DMatrix(X_test, label=y_test)
params_no_constraints = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42}
model_no_constraints = xgb.train(params_no_constraints, dtrain_no_constraints, num_boost_round=10)
predictions_no_constraints = (model_no_constraints.predict(dtest_no_constraints) > 0.5).astype(int)
accuracy_no_constraints = accuracy_score(y_test, predictions_no_constraints)

print(f"Accuracy without constraint: {accuracy_no_constraints:.4f}")
```

Here, we use two features derived from the original number of calls, one representing calls below 5 with a monotonic decreasing constraint, and one representing calls above 5 with a monotonic increasing constraint to simulate a local minimum effect. Again, the accuracy numbers are similar, but the constrained model has a far better interpretability for the call support feature.

Now, a key aspect to understand is when and when not to use them. Monotonic constraints aren't a silver bullet; they work best when you have a solid understanding of how your features should behave in relation to the target variable. If you are unsure, it's advisable to start without them and only add them when there's a good theoretical or domain reason for doing so and when it improves the interpretability of the model. Over-constraining a model can also hinder performance if these constraints are not accurate representations of reality, leading to potentially worse predictions than an unconstrained model.

For further exploration, I highly recommend delving into the original xgboost paper and the corresponding section discussing monotonic constraints. Also, the "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, and "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Géron provide a solid theoretical foundation for understanding tree-based methods and will give you additional context on the uses for monotonicity. Understanding the inner workings of tree construction, the algorithms employed by xgboost, and the effects of regularization on generalization will also be a very good starting point for experimenting with these constraints. They are tools that need both caution and careful application to extract their full value.
