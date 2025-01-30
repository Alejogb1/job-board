---
title: "How can I create prediction intervals for a neural network in Python?"
date: "2025-01-30"
id: "how-can-i-create-prediction-intervals-for-a"
---
Prediction intervals, unlike point predictions, quantify the uncertainty inherent in a neural network's output, providing a range within which a predicted value is likely to fall with a certain probability. Point predictions alone often mask the model's confidence, making them insufficient for decision-making in many real-world scenarios. Generating these intervals requires techniques that go beyond standard neural network training. I've personally encountered this challenge across several projects, notably in demand forecasting where understanding the range of possible outcomes was just as critical as the point prediction itself.

One common approach is to employ a technique based on quantile regression. Instead of training a neural network to predict the conditional *mean* of the target variable, we train it to predict specific *quantiles* of the conditional distribution. Quantiles represent the points below which a certain proportion of the data is expected to lie. For instance, the 0.05 quantile represents the value below which 5% of the data falls, and the 0.95 quantile represents the value below which 95% of the data falls. By training a network to predict several quantiles simultaneously, we can construct a prediction interval. Specifically, the interval would be bounded by a lower quantile (e.g., 0.025) and an upper quantile (e.g., 0.975), thereby approximating a 95% prediction interval. The crucial point is that the neural network is not simply learning to predict the mean; instead it's learning different parts of the entire conditional probability distribution.

Implementing quantile regression within a neural network involves a modification of the loss function. Instead of using a traditional loss like Mean Squared Error (MSE), which focuses on the conditional mean, we employ a quantile loss function. The quantile loss is defined as:

`ρ(y - ŷ) = (y - ŷ) * τ  if (y - ŷ) >= 0`

`ρ(y - ŷ) = (ŷ - y) * (1 - τ)  if (y - ŷ) < 0`

where `y` is the true value, `ŷ` is the predicted value, and `τ` is the target quantile (e.g., 0.05, 0.5, 0.95). This piecewise function ensures that if the prediction is below the target, there is a penalty proportional to the undershoot multiplied by 1-τ, whereas if the prediction is above the target, the penalty is proportional to the overshoot multiplied by τ. This asymmetric nature of the loss forces the network to learn the appropriate quantiles.

Another technique involves using Bayesian Neural Networks (BNNs). In BNNs, the model parameters (weights and biases) are treated as probability distributions, instead of fixed values. This allows us to sample from the posterior distribution of the network parameters after training, resulting in a predictive distribution, rather than a single point prediction. The spread of this predictive distribution then provides a measure of uncertainty. Although more computationally expensive than quantile regression, BNNs often give a richer and more nuanced view of the prediction uncertainty.

Here are three Python code examples demonstrating these approaches. The examples use `TensorFlow` and `scikit-learn` for clarity:

**Example 1: Quantile Regression with a Simple Neural Network**

```python
import tensorflow as tf
import numpy as np

def quantile_loss(y_true, y_pred, quantile):
    err = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * err, (quantile - 1) * err))

def build_quantile_model(input_shape, quantiles):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    outputs = [tf.keras.layers.Dense(1)(x) for _ in quantiles]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Sample data
X_train = np.random.rand(1000, 5)
y_train = np.sin(np.sum(X_train, axis=1)) + np.random.normal(0, 0.1, 1000)
X_test = np.random.rand(200, 5)


quantiles = [0.025, 0.5, 0.975]
model = build_quantile_model(X_train.shape[1], quantiles)

# Compile with a separate loss for each quantile
losses = [lambda y_true, y_pred: quantile_loss(y_true, y_pred, q) for q in quantiles]
model.compile(optimizer='adam', loss=losses)

# Training (pass a list of labels as input for each quantile)
model.fit(X_train, [y_train]*len(quantiles), epochs=100, batch_size=32, verbose=0)


# Prediction (results are a list of predictions)
predictions = model.predict(X_test)

lower_bound = predictions[0].flatten()
median = predictions[1].flatten()
upper_bound = predictions[2].flatten()

print("Lower Bound:", lower_bound[:5])
print("Median:", median[:5])
print("Upper Bound:", upper_bound[:5])
```
This code defines a neural network that predicts three quantiles: 0.025, 0.5, and 0.975, allowing the user to derive a 95% prediction interval. The `quantile_loss` function applies the previously described asymmetric loss function. The model is trained simultaneously for each quantile, and the output contains a separate prediction for each one.

**Example 2: Monte Carlo Dropout for Uncertainty Estimation**

```python
import tensorflow as tf
import numpy as np

def build_dropout_model(input_shape):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

X_train = np.random.rand(1000, 5)
y_train = np.sin(np.sum(X_train, axis=1)) + np.random.normal(0, 0.1, 1000)
X_test = np.random.rand(200, 5)

model = build_dropout_model(X_train.shape[1])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Prediction with dropout
n_samples = 100
predictions = []
for _ in range(n_samples):
    predictions.append(model(X_test, training=True).numpy().flatten()) # training=True activates dropout

predictions = np.array(predictions)
mean_prediction = np.mean(predictions, axis=0)
std_prediction = np.std(predictions, axis=0)

# Construct a prediction interval (assuming normality)
lower_bound = mean_prediction - 1.96 * std_prediction
upper_bound = mean_prediction + 1.96 * std_prediction

print("Lower Bound (dropout):", lower_bound[:5])
print("Mean (dropout):", mean_prediction[:5])
print("Upper Bound (dropout):", upper_bound[:5])
```

This example demonstrates how to use Monte Carlo dropout as an approximation for Bayesian inference. By activating dropout during prediction, we obtain multiple slightly different outputs. The distribution of these outputs can be used to estimate the uncertainty of the predictions. I have encountered situations where this method provides a good compromise between performance and uncertainty estimation.

**Example 3: Using an Ensemble Approach**

```python
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def build_ensemble_model(input_shape):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_ensemble(X, y, num_models):
    models = []
    for i in range(num_models):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model = build_ensemble_model(X.shape[1])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_val, y_val))
        models.append(model)
    return models

X_train = np.random.rand(1000, 5)
y_train = np.sin(np.sum(X_train, axis=1)) + np.random.normal(0, 0.1, 1000)
X_test = np.random.rand(200, 5)

num_models = 5
ensemble_models = train_ensemble(X_train, y_train, num_models)

ensemble_predictions = []
for model in ensemble_models:
    ensemble_predictions.append(model.predict(X_test).flatten())
ensemble_predictions = np.array(ensemble_predictions)

mean_prediction = np.mean(ensemble_predictions, axis=0)
std_prediction = np.std(ensemble_predictions, axis=0)


lower_bound = mean_prediction - 1.96 * std_prediction
upper_bound = mean_prediction + 1.96 * std_prediction


print("Lower Bound (ensemble):", lower_bound[:5])
print("Mean (ensemble):", mean_prediction[:5])
print("Upper Bound (ensemble):", upper_bound[:5])
```
This example constructs a simple ensemble model. Multiple models are trained on slightly different subsets of the training data. The variation in the predictions from each model can also serve as a measure of uncertainty. This technique proved effective in my work with chaotic time series data.

When deciding on an approach, consider the computational costs and complexity. Quantile regression is relatively simple and efficient, making it suitable for large datasets. BNNs provide more comprehensive uncertainty quantification but can be computationally expensive, and approximations like MC dropout can offer a middle ground. Ensemble methods represent another option, with their own trade-offs.

For further information on Bayesian neural networks, review research on variational inference and Markov Chain Monte Carlo methods. Material covering loss functions for quantile regression can provide a more fundamental understanding of the underlying mathematics. Finally, works on ensemble learning techniques will furnish supplementary approaches to model uncertainty quantification. These resources have all proven useful in my own work. Careful consideration of these points will enable you to effectively implement prediction intervals for neural networks.
