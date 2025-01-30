---
title: "How to display prediction probabilities in Keras?"
date: "2025-01-30"
id: "how-to-display-prediction-probabilities-in-keras"
---
The core challenge in displaying prediction probabilities from a Keras model lies in understanding the output layer's activation function and its relationship to the model's intended task.  My experience building numerous classification and regression models in Keras has consistently highlighted this crucial point: the raw output of the model isn't directly interpretable as probability unless specifically designed for that purpose.  This response will address this issue, focusing on distinct scenarios and providing clear code examples.

**1. Understanding Output Layer Activation and Model Architecture:**

The activation function of your output layer dictates the interpretation of the model's output.  For binary classification, a sigmoid activation function produces a single value between 0 and 1, representing the probability of the positive class.  For multi-class classification, a softmax activation function outputs a vector where each element represents the probability of a specific class, with the probabilities summing to 1. Regression models, conversely, typically use linear activations and their output needs post-processing to generate probabilities if required.  Failure to consider this fundamental aspect leads to incorrect probability interpretations and flawed analysis.

Misinterpreting the output is common.  For example, a binary classification model with a linear activation function will produce raw scores, not probabilities. Similarly, a multi-class model using a sigmoid for each class would generate independent probabilities that wouldn't necessarily sum to 1.  Therefore, choosing the correct activation is paramount.

**2. Code Examples:**

**Example 1: Binary Classification (Sigmoid Activation)**

This example demonstrates a binary classification model predicting the likelihood of customer churn. The sigmoid activation directly provides the probability of churn.

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Sample data (replace with your actual data)
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)

# Model definition
model = keras.Sequential([
    Dense(16, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid') # Sigmoid for probability output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Make predictions
predictions = model.predict(X_train)

# Display probabilities
for i in range(5): # Display probabilities for first five samples
    print(f"Sample {i+1}: Probability of churn = {predictions[i][0]:.4f}")

```

The `.predict()` method returns an array of probabilities directly.  The `[i][0]` indexing accesses the single probability value from each prediction.  The `:.4f` formatting ensures a clear representation of the probability with four decimal places.


**Example 2: Multi-class Classification (Softmax Activation)**

This example predicts the category of an image using a multi-class model.  Softmax ensures the output is a probability distribution over all classes.

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Sample data (replace with your actual image data)
X_train = np.random.rand(100, 784)  # 100 samples, 784 features (e.g., flattened images)
y_train = keras.utils.to_categorical(np.random.randint(0, 10, 100), num_classes=10) # 10 classes

# Model definition
model = keras.Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax') # Softmax for probability distribution
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Make predictions
predictions = model.predict(X_train)

# Display probabilities
for i in range(5):
    print(f"Sample {i+1}: Probabilities = {predictions[i]}")
    predicted_class = np.argmax(predictions[i])
    print(f"Predicted class: {predicted_class}")

```
Here, `keras.utils.to_categorical` transforms integer labels into one-hot encoded vectors, essential for categorical cross-entropy loss. The `softmax` activation transforms the model's output into a probability distribution across the ten classes.  The `np.argmax()` function identifies the class with the highest probability.

**Example 3: Regression with Probability Calibration (Sigmoid for Post-processing)**

Regression models, such as those predicting continuous variables like house prices, don't directly output probabilities.  However, we can calibrate the output using a sigmoid function to generate probabilistic predictions within a specific range. This example assumes we want to predict the probability of a house price exceeding a certain threshold.

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from scipy.special import expit #More numerically stable sigmoid

# Sample data (replace with your actual data)
X_train = np.random.rand(100, 3)  # 100 samples, 3 features
y_train = np.random.rand(100) * 1000000 #House prices

# Define threshold
threshold = 500000

# Model definition (linear activation for regression)
model = keras.Sequential([
    Dense(32, activation='relu', input_shape=(3,)),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse') # Mean squared error for regression

# Train the model
model.fit(X_train, y_train, epochs=10)

# Make predictions and calibrate to probabilities
predictions = model.predict(X_train)
probabilities = expit((predictions - threshold) / 100000) #Calibration using sigmoid


# Display probabilities of exceeding the threshold
for i in range(5):
    print(f"Sample {i+1}: Probability of exceeding {threshold} = {probabilities[i][0]:.4f}")
```

In this regression scenario,  we first make predictions using the linear model. Subsequently, we apply a sigmoid function (using `expit` for numerical stability) to the difference between predictions and the threshold, scaling appropriately to produce interpretable probabilities.  This calibration step allows us to interpret the model's output within a probabilistic framework, suitable for decision-making.


**3. Resource Recommendations:**

The Keras documentation provides comprehensive details on model building, activation functions, and loss functions.  Understanding probability distributions and statistical concepts will enhance your ability to interpret model outputs correctly.  Exploring different optimization algorithms and their impact on model performance is also crucial.  Finally, learning about model evaluation metrics relevant to classification and regression problems is essential for reliable results.
