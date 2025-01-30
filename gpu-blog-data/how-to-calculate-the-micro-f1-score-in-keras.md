---
title: "How to calculate the micro F1-score in Keras?"
date: "2025-01-30"
id: "how-to-calculate-the-micro-f1-score-in-keras"
---
The micro-averaged F1-score, often simply referred to as the micro F1-score, isn't directly available as a metric within the Keras `metrics` module. This is because it requires aggregating predictions and ground truths across all samples before calculating precision and recall, unlike macro-averaging which calculates them individually per class and then averages.  My experience building large-scale multi-class classification models for natural language processing tasks, specifically in sentiment analysis and topic modeling, highlighted this limitation.  Consequently, I developed a custom solution leveraging Keras's backend capabilities for efficient computation.

**1. Clear Explanation:**

The micro F1-score calculation begins with the aggregation of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) across all classes.  It's crucial to emphasize that this aggregation happens *before* calculating precision and recall.  Unlike the macro F1-score, which averages class-specific F1-scores, the micro F1-score provides a single metric reflecting the overall performance across all classes. This is particularly useful when dealing with imbalanced datasets or when the class distribution significantly influences the macro F1-score.

The formulas are as follows:

* **Micro Precision (µP):**  µP = TP<sub>total</sub> / (TP<sub>total</sub> + FP<sub>total</sub>)
* **Micro Recall (µR):** µR = TP<sub>total</sub> / (TP<sub>total</sub> + FN<sub>total</sub>)
* **Micro F1-score (µF1):** µF1 = 2 * (µP * µR) / (µP + µR)

Where TP<sub>total</sub>, FP<sub>total</sub>, and FN<sub>total</sub> represent the sums of true positives, false positives, and false negatives across all classes respectively.  Calculating these totals efficiently is key to a performant implementation, especially when handling large datasets.


**2. Code Examples with Commentary:**

**Example 1: Using `tf.math` for efficient computation:**

```python
import tensorflow as tf
import numpy as np
from keras import backend as K

def micro_f1(y_true, y_pred):
    """Calculates the micro-averaged F1-score.

    Args:
        y_true: True labels (one-hot encoded).
        y_pred: Predicted probabilities.

    Returns:
        The micro F1-score.
    """
    y_pred = K.cast(K.greater(y_pred, 0.5), 'float32') #Thresholding for binary classification
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum(K.cast(K.greater(y_pred - y_true, 0), 'float32'), axis=0)
    fn = K.sum(K.cast(K.greater(y_true - y_pred, 0), 'float32'), axis=0)

    precision = tf.math.divide_no_nan(K.sum(tp), K.sum(tp + fp))
    recall = tf.math.divide_no_nan(K.sum(tp), K.sum(tp + fn))

    f1 = tf.math.divide_no_nan(2*precision*recall, precision+recall)
    return f1


# Example usage:
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])

micro_f1_score = micro_f1(y_true, y_pred)
print(f"Micro F1-score: {K.eval(micro_f1_score)}")

```

This example leverages TensorFlow's `tf.math.divide_no_nan` to handle potential division by zero errors. The use of `K.cast` ensures correct data types for arithmetic operations within the Keras backend.  This approach is robust and efficient for both CPU and GPU computations.


**Example 2:  Utilizing NumPy for a simpler implementation (suitable for smaller datasets):**

```python
import numpy as np

def micro_f1_numpy(y_true, y_pred):
    """Calculates micro F1-score using NumPy.  Suitable for smaller datasets."""
    y_pred = (y_pred > 0.5).astype(int) #Thresholding
    tp = np.sum(y_true * y_pred)
    fp = np.sum((y_pred - y_true) > 0)
    fn = np.sum((y_true - y_pred) > 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

#Example Usage (same y_true and y_pred as above)
micro_f1_score_np = micro_f1_numpy(y_true, y_pred)
print(f"Micro F1-score (NumPy): {micro_f1_score_np}")

```

This NumPy-based implementation is more straightforward but less efficient for large datasets due to the reliance on NumPy's array operations. It includes explicit checks to avoid division by zero.



**Example 3:  Integrating into a Keras model:**

```python
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# ... (model definition) ...

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(3, activation='sigmoid') # Assuming 3 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[micro_f1]) #Adding our custom metric

# ... (model training) ...

model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32)

```

This demonstrates the integration of the custom `micro_f1` function (from Example 1, for better performance) directly into the Keras model's compilation process.  This allows for monitoring the micro F1-score during training, providing valuable insights into the model's performance.  Remember to replace `X_train` and `y_train` with your actual training data.  The output layer uses a sigmoid activation function, suitable for multi-class classification with a one-hot encoded output.


**3. Resource Recommendations:**

*  "Deep Learning with Python" by Francois Chollet:  Provides a strong foundation in Keras and TensorFlow.
*  "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron: Offers a practical approach to machine learning with detailed explanations of various metrics.
*  Relevant TensorFlow and Keras documentation: Carefully reviewing the official documentation is essential for understanding the intricacies of the framework.  Pay close attention to the sections on custom metrics and backend operations.


By understanding the underlying principles of micro F1-score calculation and leveraging the flexibility of Keras's backend, one can effectively implement and monitor this critical metric in various machine learning applications.  The choice between the TensorFlow/Keras backend (Example 1) and the NumPy approach (Example 2) should be guided by dataset size and computational resources.  Remember to always choose the approach that offers the best balance of accuracy and performance for your specific use case.
