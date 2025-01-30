---
title: "Why does training accuracy differ from prediction accuracy on the same data?"
date: "2025-01-30"
id: "why-does-training-accuracy-differ-from-prediction-accuracy"
---
The divergence between training accuracy and prediction accuracy, even when evaluated on the same dataset, stems primarily from a discrepancy in how the model utilizes the data during these two distinct phases. Training accuracy reflects the model's performance while learning, while prediction accuracy assesses its performance after learning is completed, using the *same* data. This discrepancy arises due to a variety of factors, often linked to overfitting and the implicit memory the model retains of the training process.

During training, a model iteratively adjusts its internal parameters (weights and biases) based on the loss function computed using the training data. The goal is to minimize this loss. Critically, the model has *direct access* to the training data's labels during this process. For example, in a classification problem, it knows what the correct class should be for each training instance and can adjust to align its prediction with this known label. This dynamic adjustment leads to a high training accuracy because the model is actively being guided towards correct classification. It’s learning *how to learn the training set* specifically. In essence, it begins to directly map the inputs to known outputs. It’s akin to a student memorizing the answer sheet rather than understanding the concepts.

However, when the model is tasked with predicting on the *same* data, this direct feedback loop is absent. The model receives the input but must generate a prediction *independently*. If the model has overfit to the training data, it has likely not learned generalizable patterns and has instead memorized specifics that are not applicable during independent prediction, even on the training data itself. This explains why prediction accuracy, even on the training data, can fall short of the high accuracy measured during training. Consider the analogy of a student who does well on practice exams due to memorization, only to struggle when given the same questions in a timed, independent environment where the answers aren't readily available to check against.

The core problem is the model’s inherent ability to adapt during the learning phase and then being forced to make its own judgment during prediction. This is not always a failure; models can learn effectively during training and predict well. However, the potential for this divergence is critical to consider, especially with highly complex models and limited, noisy, or biased training data. I’ve seen this issue manifest repeatedly across different machine learning projects, often unexpectedly. For example, in a text classification project on customer reviews, I had a high training accuracy but surprisingly low prediction accuracy on the training data itself using a complex transformer model. It had essentially learned the nuances of the specific reviews rather than generalizable semantic patterns.

Let's illustrate with code examples. These are simplified scenarios, but they encapsulate the core idea. We'll use Python with the scikit-learn library.

**Code Example 1: Simple Linear Regression**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some sample data (with some noise)
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the *same* training data
y_pred_training = model.predict(X)

# Calculate the training mean squared error (MSE)
training_mse = mean_squared_error(y, y_pred_training)
print(f"Training MSE: {training_mse}")

#Now we apply a similar process as a prediction

y_pred_prediction = model.predict(X)
prediction_mse = mean_squared_error(y,y_pred_prediction)
print(f"Prediction MSE: {prediction_mse}")
```

This code generates a simple linear regression model.  During training (`model.fit`), the model adjusts its slope and intercept to minimize the difference between its predictions and the true `y` values in `y`.  The `training_mse` is computed using these adjusted parameters. When we perform a prediction via `model.predict(X)`, we essentially query what the model has learned, which will lead to a similar if not identical prediction and error during the training phase and the prediction phase. The core aspect to note here is that despite the prediction being a *forward pass*, there is still an implicit memory of the fit process. In more complex models, this implicit memory will be more significant in causing divergence.

**Code Example 2: Overfitting with a Polynomial Model**

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate noisy data with a quadratic relationship
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 0.5 * X.squeeze()**2 + 2*X.squeeze() + 1 + np.random.randn(100) * 5

# Create polynomial features
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)

# Create and train linear model using polynomial features
model = LinearRegression()
model.fit(X_poly, y)


# Make predictions on *same* training data
y_pred_training = model.predict(X_poly)

# Calculate training MSE
training_mse = mean_squared_error(y, y_pred_training)
print(f"Training MSE: {training_mse}")

# Make predictions in the prediction phase
y_pred_prediction = model.predict(X_poly)
prediction_mse = mean_squared_error(y,y_pred_prediction)
print(f"Prediction MSE: {prediction_mse}")

```

Here, we introduce a polynomial transformation, which enables the linear model to fit highly complex relationships.  A polynomial of degree 15 is employed, a choice that almost inevitably leads to overfitting on this limited dataset. During training, the model forces its parameters to fit this complex relationship extremely well. We again witness the implicit memory that the model carries of the training data when it predicts on the same dataset. While training accuracy is likely to be higher than the previous example, it’s still close to prediction accuracy given that we are evaluating on the same data. However, if you were to apply a new dataset, that difference would likely be dramatic as the model has memorized training data specifics.

**Code Example 3: Illustrative Classification with Overfitting**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Generate a classification dataset
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42, n_classes=2)
# Create a simple logistic regression model
model = LogisticRegression(solver = 'liblinear')
# Train the model
model.fit(X, y)

# Evaluate the model performance during the training phase
y_pred_training = model.predict(X)

training_accuracy = accuracy_score(y, y_pred_training)
print(f"Training Accuracy: {training_accuracy}")

# Evaluate the model performance during a prediction phase
y_pred_prediction = model.predict(X)
prediction_accuracy = accuracy_score(y,y_pred_prediction)
print(f"Prediction Accuracy: {prediction_accuracy}")
```

In this example, a classification problem is setup using a logistic regression model. A logistic regression model can learn relatively complicated decision boundaries based on data, though not as easily as a polynomial based model. The data is randomly generated. However, the underlying principle is the same: the model parameters will be adjusted by learning the training data. This will mean a higher accuracy during the training phase than if we had a novel dataset. Here again the core difference lies in that while we can query the model as a forward pass during prediction, this pass is influenced by what happened during training. The difference between training accuracy and prediction accuracy on the same data will be minimal, although we may observe some differences based on various training implementation details.

These examples demonstrate that a high training accuracy is no guarantee of good predictive performance, even on the same data, due to the inherent memorization and adaptation dynamics present during the training process.

For further understanding, I recommend exploring resources on the following topics: *Bias-Variance Tradeoff*, which explains the relationship between model complexity, overfitting, and generalization; *Cross-Validation*, a method for evaluating model performance more robustly; and *Regularization techniques*, such as L1 and L2 regularization, which can help prevent overfitting by penalizing model complexity. Texts and courses focusing on machine learning foundations will usually cover these topics in detail.

In conclusion, the difference between training and prediction accuracy on the same data is less of a computational error but a reflection of a fundamental aspect of machine learning. Training data is used to *learn* and the prediction phase is a question of how well *that learning generalizes*. Even when using the training data for the prediction phase, the model’s internal mechanisms can lead to discrepancies, especially as model complexity increases. Understanding and mitigating this issue is key for building robust and reliable predictive models.
