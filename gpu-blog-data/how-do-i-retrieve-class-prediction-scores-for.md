---
title: "How do I retrieve class prediction scores for all classes after running `predict_classes`?"
date: "2025-01-30"
id: "how-do-i-retrieve-class-prediction-scores-for"
---
The `predict_classes` method in Keras, while seemingly convenient, often obscures the underlying probability distributions which are essential for nuanced analysis. It directly returns the class with the highest probability, not the full set of per-class prediction scores, making post-processing and detailed evaluations cumbersome. I've encountered this limitation numerous times while fine-tuning deep learning models for medical image analysis, where access to the probability vector is crucial for uncertainty estimation and subsequent clinical decision support.

To obtain the class prediction scores for all classes, you must use the `predict` method rather than `predict_classes`. The `predict` method, applied to a model instance, returns a NumPy array representing the predicted probabilities for each class. This output is a multi-dimensional array where the first dimension usually represents the batch size and the second dimension, the number of classes. Each element in this output is a floating-point value between 0 and 1, representing the probability of a given sample belonging to the corresponding class. These values always sum to 1 across all classes for any given instance.

Let’s clarify with practical examples using a hypothetical Keras model. Assume we have a model trained for a multiclass classification problem with, let's say, five classes.

**Code Example 1: Basic Prediction with `predict`**

```python
import numpy as np
from tensorflow import keras

# Hypothetical model (replace with your actual model)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(5, activation='softmax')
])

# Sample input data (replace with your data)
sample_data = np.random.rand(1, 10) # Batch of 1 sample, 10 features

# Making predictions using 'predict'
predictions = model.predict(sample_data)

# The 'predictions' array is the key here.
print("Shape of predictions array:", predictions.shape)
print("Prediction probabilities:", predictions)
```

In this example, the `model.predict(sample_data)` call returns a NumPy array of shape `(1, 5)`. The first dimension, in this case, is of size one because we provided one sample for prediction. The second dimension of size 5 corresponds to the five classes in our model's output. The elements within the returned array hold the probability scores for each respective class. Each element within the nested array is a probability, ranging from zero to one, and the sum across these probabilities will always equal one. This approach yields precisely the class probabilities we need, rather than just a single most likely class label as returned by `predict_classes`.

**Code Example 2: Extracting Probabilities for Multiple Samples**

```python
import numpy as np
from tensorflow import keras

# Hypothetical model (replace with your actual model)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(5, activation='softmax')
])

# Multiple sample data
sample_data = np.random.rand(10, 10)  # Batch of 10 samples, 10 features

# Making predictions
predictions = model.predict(sample_data)

print("Shape of predictions array:", predictions.shape)
print("First sample probabilities:", predictions[0,:])
print("Fifth sample probabilities:", predictions[4,:])
```

Here, the input data consists of 10 samples, resulting in a `predictions` array of shape `(10, 5)`. Now each row represents the probability distribution for a different input sample. For instance, `predictions[0,:]` returns the probability distribution for the first sample across all five classes. Indexing the `predictions` array, allows us to conveniently retrieve the probability vector for any sample. This is crucial in real-world applications. When working with datasets from my experience in anomaly detection within industrial IoT sensors, I often needed to analyze the probability trends across a large number of sequential data points. Being able to quickly retrieve the full probabilities for each sample made it significantly easier to identify anomalies based on low-confidence predictions or shifting class probabilities.

**Code Example 3: Converting to Class Labels and Probabilities**

```python
import numpy as np
from tensorflow import keras

# Hypothetical model (replace with your actual model)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(5, activation='softmax')
])

# Multiple samples
sample_data = np.random.rand(10, 10)

# Predictions
predictions = model.predict(sample_data)

# Getting predicted class labels
predicted_classes = np.argmax(predictions, axis=1)

# Printing
print("Shape of predictions array:", predictions.shape)
print("Predicted probabilities:", predictions)
print("Predicted Classes:", predicted_classes)

for i in range(len(predicted_classes)):
    print(f"Sample {i+1}, Predicted Class: {predicted_classes[i]}, Probabilities: {predictions[i,:]}")
```

This example illustrates the common workflow of first obtaining the full probability vector using `predict` and then transforming those probabilities into a single class prediction. The `np.argmax(predictions, axis=1)` function is applied to find the index of the maximum probability across all classes for each sample.  This effectively replicates what `predict_classes` does, but it does it *after* we've retrieved the full probability vector. Notice how both the predicted class label as well as all of the predicted probabilities are available for examination. This level of access to predicted probability scores has been critical in the model debugging I've done. For example, if a model is consistently misclassifying a certain type of input, it’s useful to examine if the confidence scores for the true class are unusually low, revealing potentially overlooked features.

In practical terms, the flexibility afforded by accessing full class probability distributions greatly enhances model interpretability and opens doors to more advanced post-processing techniques. For instance, you can use these probabilities for:

*   **Thresholding:** Applying different thresholds to the probability scores, allowing more nuanced classification decisions, which can be critical in high-stakes applications where the cost of misclassification varies.
*   **Ensemble Methods:** Combining probabilities from multiple models for improved performance. The individual predictions from models trained on different sub-datasets, are often combined through averaging or other methods. Accessing the full vector allows for this combination.
*   **Uncertainty Quantification:** Analyzing the probability distribution to quantify model uncertainty, which is crucial in situations where you require the model to indicate when it’s not confident in its prediction.
*   **Model Calibration:** Checking and adjusting if the model outputs calibrated probabilities, where probability scores accurately represent the true likelihood of a class.

For further information, I would recommend exploring documentation provided by the TensorFlow and Keras teams regarding model prediction. Also relevant are resources focused on model evaluation, particularly concepts like precision, recall, and the area under the ROC curve. Additionally, publications detailing specific classification model types like logistic regression, convolutional neural networks and transformer architectures often discuss the interpretation of probability scores within the context of their associated algorithms.
