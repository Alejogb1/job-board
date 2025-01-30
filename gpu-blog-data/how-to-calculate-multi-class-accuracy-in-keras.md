---
title: "How to calculate multi-class accuracy in Keras?"
date: "2025-01-30"
id: "how-to-calculate-multi-class-accuracy-in-keras"
---
The inherent challenge in calculating multi-class accuracy within the Keras framework stems from its reliance on categorical predictions rather than single-valued outputs.  Directly applying binary accuracy metrics will yield incorrect results, necessitating a more nuanced approach.  My experience working on large-scale image classification projects highlighted this issue repeatedly.  Overcoming it involved a thorough understanding of both Keras's output format and the underlying principles of multi-class classification evaluation.

**1. Clear Explanation**

Multi-class classification problems involve assigning instances to one of *N* distinct classes.  Unlike binary classification (two classes), simple accuracy – the ratio of correctly classified instances to the total number of instances – requires careful handling in a multi-class context.  Keras, by default, outputs prediction probabilities for each class in the form of a one-hot encoded vector or a probability distribution.  Therefore, we cannot simply compare the raw output to the true labels; instead, we must first identify the predicted class based on the maximum probability and then compare this with the true class label.

The accuracy calculation involves these steps:

1. **Prediction:**  The Keras model generates a prediction for each instance, typically a probability vector of length *N* where *N* is the number of classes.

2. **Class Identification:** The class with the highest probability is identified as the predicted class for each instance.  This typically involves using `np.argmax()`.

3. **Comparison:** The predicted class is compared to the true class label for each instance.  A match results in a correct classification.

4. **Accuracy Calculation:** The overall accuracy is calculated as the ratio of correctly classified instances to the total number of instances.

This process can be implemented using various approaches, including custom metrics, leveraging Keras built-in functions, or using NumPy for post-processing of model outputs.  I've found each approach valuable in different contexts depending on project requirements and desired level of integration with the training process.


**2. Code Examples with Commentary**

**Example 1: Custom Metric Function**

This approach provides the most direct integration with the Keras training process.  It allows for real-time monitoring of accuracy during training and avoids post-processing steps.

```python
import tensorflow as tf
import numpy as np

def multiclass_accuracy(y_true, y_pred):
  """Calculates multi-class accuracy.

  Args:
    y_true: True labels (one-hot encoded).
    y_pred: Predicted probabilities.

  Returns:
    The multi-class accuracy.
  """
  y_pred_classes = tf.argmax(y_pred, axis=1)
  y_true_classes = tf.argmax(y_true, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(y_pred_classes, y_true_classes), tf.float32))

# Example usage:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[multiclass_accuracy])
```

This code defines a custom metric function `multiclass_accuracy` which takes the true labels (`y_true`) and predicted probabilities (`y_pred`) as input. It uses `tf.argmax` to find the predicted class and then compares it to the true class using `tf.equal`. Finally, it calculates the mean accuracy using `tf.reduce_mean`.  In my experience, this method provided the cleanest integration with the Keras training loop, offering instantaneous feedback on accuracy during model development.


**Example 2: Post-processing with NumPy**

This method involves obtaining predictions after model training and then calculating accuracy using NumPy.  This approach is straightforward but lacks the real-time feedback of the custom metric function.

```python
import numpy as np
from sklearn.metrics import accuracy_score

# ... model training ...

y_true = np.argmax(y_test, axis=1) # Assuming y_test is one-hot encoded
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"Multi-class accuracy: {accuracy}")
```

Here, we use `model.predict` to get predictions on a test set and then use `np.argmax` to convert both true and predicted labels to class indices.  The `accuracy_score` function from scikit-learn directly computes the accuracy.  This approach is particularly useful when evaluating models after training, potentially utilizing other scikit-learn metrics for a comprehensive evaluation. I used this extensively for generating comprehensive reports after model training and hyperparameter tuning.


**Example 3: Using `sparse_categorical_accuracy`**

If your true labels are not one-hot encoded but are instead integer class indices, Keras provides a built-in metric that simplifies the calculation.

```python
import tensorflow as tf

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#Note: y_true should now be an array of integers representing class indices.
```

This leverages Keras's built-in `sparse_categorical_accuracy` metric. The key difference is the use of `sparse_categorical_crossentropy` as the loss function, and the true labels (`y_true`) are now integer class labels, not one-hot encoded vectors.  This option significantly simplifies the code, especially when working with datasets already formatted with integer labels, a common situation encountered when dealing with large datasets.  This method was particularly efficient when importing datasets with pre-defined labels from external sources.


**3. Resource Recommendations**

The Keras documentation, particularly the sections on metrics and loss functions, provides detailed information on available options and their usage.   A solid understanding of linear algebra and probability is crucial for interpreting model outputs and choosing appropriate evaluation metrics.  Furthermore, studying various machine learning textbooks covering multi-class classification and model evaluation will significantly enhance your understanding of these concepts.  Familiarizing oneself with the scikit-learn library is also highly beneficial for performing comprehensive model evaluations, encompassing accuracy and other relevant performance indicators.
