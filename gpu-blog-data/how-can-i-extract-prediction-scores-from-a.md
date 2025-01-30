---
title: "How can I extract prediction scores from a Keras MLP inference?"
date: "2025-01-30"
id: "how-can-i-extract-prediction-scores-from-a"
---
The crucial point regarding prediction score extraction from a Keras MLP inference lies in understanding that the output layer's activation function dictates the format of the predictions.  Unlike classification problems where a softmax activation provides probabilities directly, regression and other tasks require explicit handling to obtain meaningful "prediction scores."  My experience debugging similar issues on large-scale customer churn prediction models has underscored the importance of carefully examining this layer.  Incorrectly interpreting the raw output can lead to flawed downstream analysis and ultimately, incorrect business decisions.


**1. Clear Explanation:**

A Keras Multilayer Perceptron (MLP) is a feedforward neural network.  The final layer's output represents the model's prediction.  However, the interpretation of this output varies significantly depending on the problem type and the chosen activation function.

* **Regression:**  For regression tasks (predicting continuous values), the output layer typically uses a linear activation (no activation function or `linear` in Keras).  The raw output directly represents the predicted value.  In this case, the "prediction score" can be the predicted value itself, or a derived metric such as the absolute or squared error compared to the true value.

* **Binary Classification:** For binary classification (predicting one of two classes), the output layer usually has a single neuron with a sigmoid activation.  The output represents the probability of belonging to the positive class (typically ranging from 0 to 1). This probability can directly serve as the prediction score.

* **Multi-class Classification:**  In multi-class classification (predicting one of multiple classes), the output layer usually has multiple neurons, one for each class, with a softmax activation.  The output for each neuron represents the probability of belonging to the corresponding class.  The highest probability can be taken as the prediction score, indicating the predicted class and its confidence level.  Alternatively, all probabilities can constitute the prediction scores, allowing for further analysis of uncertainty.

To extract these prediction scores, you need to access the model's output during inference.  The specific method involves using the `model.predict()` method and then processing the results based on the activation and problem type. The following sections detail this with code examples.


**2. Code Examples with Commentary:**

**Example 1: Regression (House Price Prediction)**

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Define a simple MLP for regression
model = keras.Sequential([
    Dense(64, activation='relu', input_shape=(10,)), # 10 input features
    Dense(32, activation='relu'),
    Dense(1) # Linear activation (default) for regression
])

model.compile(optimizer='adam', loss='mse') # Mean Squared Error for regression

# Sample data (replace with your own)
X_test = np.random.rand(100, 10)
y_test = np.random.rand(100, 1)

# Make predictions
predictions = model.predict(X_test)

# Predictions are the raw output -  a numpy array of predicted house prices
print(predictions)

#Calculate Mean Absolute Error as a prediction score metric
mae = np.mean(np.abs(predictions - y_test))
print(f"Mean Absolute Error: {mae}")
```

This example demonstrates a regression task. The output layer has no activation function (implicitly linear), providing direct predictions. The Mean Absolute Error (MAE) is calculated as a prediction score metric to assess the model's performance.  Replacing the loss function and MAE calculation provides flexibility to use other regression performance metrics.


**Example 2: Binary Classification (Spam Detection)**

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Define a simple MLP for binary classification
model = keras.Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Sample data (replace with your own)
X_test = np.random.rand(100, 10)
y_test = np.random.randint(0, 2, 100) # 0 or 1 for binary classification

# Make predictions
predictions = model.predict(X_test)

# Predictions are probabilities - a numpy array of probabilities between 0 and 1
print(predictions)

# Classification based on threshold (e.g., 0.5)
predicted_classes = (predictions > 0.5).astype(int)

# Calculate accuracy or other binary classification metrics
accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy: {accuracy}")

```

Here, a sigmoid activation produces probabilities between 0 and 1. These probabilities are the prediction scores,  representing the model's confidence in classifying an email as spam or not spam. A threshold of 0.5 is used to classify examples, and accuracy is used as an evaluation metric.  Precision, recall, or F1-score could be used instead.


**Example 3: Multi-class Classification (Image Classification)**

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Define a simple MLP for multi-class classification
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax') # Softmax for multi-class classification (10 classes)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Sample data (replace with your own)
X_test = np.random.rand(100, 100)
y_test = keras.utils.to_categorical(np.random.randint(0, 10, 100), num_classes=10)

# Make predictions
predictions = model.predict(X_test)

# Predictions are probability distributions - a numpy array of probabilities for each class
print(predictions)

# Predicted class is the index of the maximum probability
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy or other multi-class classification metrics
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy: {accuracy}")

```

This example utilizes softmax activation, resulting in a probability distribution over 10 classes. The prediction scores are these probabilities, and the class with the highest probability is chosen as the predicted class.  Accuracy is used as a metric, but other metrics like macro-averaged precision, recall, or F1-score are more suitable for imbalanced datasets.



**3. Resource Recommendations:**

For a deeper understanding of Keras, I recommend consulting the official Keras documentation.  The TensorFlow documentation also provides extensive resources on model building and evaluation.  A thorough grounding in linear algebra and probability theory is crucial for interpreting the output of neural networks effectively.  Furthermore, exploring texts on machine learning and deep learning will further enhance your capabilities in this domain.  Finally,  familiarizing oneself with various metrics for evaluating model performance is vital for extracting meaningful insights from prediction scores.
