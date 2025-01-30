---
title: "Why is training loss significantly lower than test loss with identical data?"
date: "2025-01-30"
id: "why-is-training-loss-significantly-lower-than-test"
---
The discrepancy between training and test loss, even when using identical data, stems primarily from the fact that a model’s optimization process directly minimizes loss on the training set, creating a bias towards that specific instance of the data while aiming to generalize on unseen data that the model does not experience during training. This discrepancy manifests when a learning algorithm overfits to random noise patterns or features specific to the training data, failing to capture the true underlying distribution governing the data.

Specifically, the “identical data” condition highlights a common misunderstanding. While the datasets themselves are comprised of the same samples, they are employed in distinctly separate roles within the model training and evaluation cycle, leading to two markedly different outcomes. The training phase uses the data to adjust model weights, pushing the model towards a low loss state specific to that training split. Conversely, test loss is calculated without any weight adjustments, serving as an independent measure of how the model generalizes to the data it was not fit to, even if this test set includes the same data instances used for training, just evaluated out-of-sample.

During training, gradients are computed solely on the training set. These gradients guide the optimizer to iteratively update model parameters, thereby reducing the loss function calculated on those exact same data samples. Consequently, the model's parameters become highly tuned to reduce loss on the training set, often memorizing the specifics of that dataset rather than learning generalizable patterns. In effect, the model constructs an “inductive bias” that perfectly fits the specific training instance, effectively forming a highly specialized, non-general solution. This is akin to memorizing the answers to a practice exam instead of learning the fundamental concepts being tested.

When the exact training set is employed as the testing set, the model is no longer being tested on data “unseen” during training. Instead, the model is tested on a dataset it has already meticulously adjusted its parameters to fit. This evaluation scenario no longer reveals how well the model will generalize to new, unseen examples drawn from the same underlying data distribution. The test loss is still measuring the error, but in the context of an instance which the model has already ‘solved’. The crucial point here is the distinction between fitting and evaluating the same dataset on training and test phases.

To illustrate, consider training a binary classifier on a small dataset containing two features, X and Y, and a binary label ‘Z’. Let us assume that the dataset, `data`, is partitioned into two identical subsets `training_data` and `testing_data`. Each set will be used separately and in a non-overlapping manner, but containing same instances as `data`.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Synthetic data generation
np.random.seed(42)
X = np.random.rand(100, 2)
Z = (X[:, 0] + X[:, 1] > 1).astype(int)

data = np.column_stack((X, Z))
training_data = data.copy()
testing_data = data.copy()

# Separate features and target
X_train = training_data[:, :2]
Z_train = training_data[:, 2]
X_test = testing_data[:, :2]
Z_test = testing_data[:, 2]

# Model instantiation and training
model = LogisticRegression()
model.fit(X_train, Z_train)

# Loss calculation
train_loss = log_loss(Z_train, model.predict_proba(X_train))
test_loss = log_loss(Z_test, model.predict_proba(X_test))

print(f"Training loss: {train_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

In the above example, the training and test data are identical copies of the generated data. The model is fitted on the training_data and then evaluated on the very same data. We should expect both training and test loss to be near similar, but the test loss is measured on data that was already used to influence model training. Therefore, the test result may not represent the real-world performance, and this practice should be avoided when conducting experiments.

A second example will further illustrate the over-fitting issue by training a model with random data. In this instance, we create features entirely independent of the labels we want to predict:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Synthetic data generation with random features
np.random.seed(42)
X = np.random.rand(100, 2)
Z = np.random.randint(0, 2, 100)

data = np.column_stack((X, Z))
training_data = data.copy()
testing_data = data.copy()

# Separate features and target
X_train = training_data[:, :2]
Z_train = training_data[:, 2]
X_test = testing_data[:, :2]
Z_test = testing_data[:, 2]

# Model instantiation and training
model = LogisticRegression()
model.fit(X_train, Z_train)

# Loss calculation
train_loss = log_loss(Z_train, model.predict_proba(X_train))
test_loss = log_loss(Z_test, model.predict_proba(X_test))

print(f"Training loss: {train_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

Here, despite X and Z being completely random, the model still manages to achieve a surprisingly low training loss. This demonstrates overfitting on noise within the training data. The testing loss is still evaluated on the same data that has been already fit to, therefore we observe a smaller test loss. Note that it does not mean the model will generalize to unseen data. The training process here is, by nature, optimized on these specific (and random) instances. This example highlights the danger of training and evaluating on identical datasets.

Finally, let’s explore the effect of increasing model capacity or complexity by adding polynomial features, and further emphasize the potential of overfitting. This will show how overfitting can decrease training loss at the expense of poor generalization.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss

# Synthetic data generation
np.random.seed(42)
X = np.random.rand(100, 2)
Z = (X[:, 0] + X[:, 1] > 1).astype(int)

data = np.column_stack((X, Z))
training_data = data.copy()
testing_data = data.copy()

# Separate features and target
X_train = training_data[:, :2]
Z_train = training_data[:, 2]
X_test = testing_data[:, :2]
Z_test = testing_data[:, 2]

# Transform to polynomial features
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model instantiation and training
model = LogisticRegression(solver='liblinear', penalty=None)
model.fit(X_train_poly, Z_train)

# Loss calculation
train_loss = log_loss(Z_train, model.predict_proba(X_train_poly))
test_loss = log_loss(Z_test, model.predict_proba(X_test_poly))

print(f"Training loss: {train_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

Here, we have added polynomial features, drastically increasing the model capacity, and without any regularization we see that the model can achieve a very low training loss. However, since we are evaluating on the same training set, we will inevitably see a smaller test loss as well. This demonstrates that even though we have identical training and test sets, complex models can still overfit and show deceptively low test losses because the evaluation is not done on a genuinely unseen test dataset.

To mitigate the problem of overfitting and address the discrepancy between training and test loss when these are indeed separate sets of different instances, several strategies can be adopted. These include the incorporation of regularization techniques like L1 or L2 regularization which impose penalties on large model weights and prevent models from memorizing noise in the training set. Another method is early stopping, where training is halted when a model’s performance on a validation set starts to degrade, thereby avoiding the later stages of training where overfitting tends to occur. Another robust approach is cross-validation, where the dataset is repeatedly partitioned into training and validation splits, thereby ensuring that the model's performance is evaluated on several subsets of the data, giving a more robust measure of generalisation. The concept of data augmentation can also be employed, to artificially increase the size of training data with the aim to increase model’s robustness.

To further develop one's understanding on this, I recommend studying books related to: Machine Learning, such as “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman, which provides a deep mathematical foundation and exploration of machine learning concepts; Deep Learning, such as “Deep Learning” by Goodfellow, Bengio, and Courville, a comprehensive overview of deep learning methodologies and theory; and related topics in Statistical Inference. Such exploration will prove invaluable in understanding the subtle nuances that lead to a discrepancy between training and test losses and techniques to mitigate these issues.
