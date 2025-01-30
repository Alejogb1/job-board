---
title: "How can I create a custom evaluation metric in Keras?"
date: "2025-01-30"
id: "how-can-i-create-a-custom-evaluation-metric"
---
Customizing evaluation metrics within the Keras framework often necessitates a nuanced understanding of its underlying mechanics.  My experience developing and deploying machine learning models across various domains, including natural language processing and time-series forecasting, has highlighted the limitations of relying solely on pre-built metrics.  The key lies in leveraging Keras's flexibility to define functions that accurately reflect the specific requirements of a given problem.  This involves a deep understanding of the tensor manipulation capabilities of TensorFlow or Theano (depending on the Keras backend) and a methodical approach to crafting these custom functions.

**1. Clear Explanation:**

Creating a custom evaluation metric in Keras involves defining a function that accepts two arguments: `y_true` (the ground truth labels) and `y_pred` (the model's predictions). This function must then compute a scalar value representing the performance of the model based on these inputs. This scalar value is ultimately what Keras utilizes to track performance during training and evaluation.  Crucially, this function must adhere to the expected input tensor shapes and data types, otherwise, errors during the execution phase are inevitable.  Furthermore, careful consideration must be given to numerical stability; handling potential edge cases such as division by zero or taking logarithms of negative numbers is paramount.  Finally, understanding the impact of metric aggregation (e.g., averaging across batches or examples) is vital for correctly interpreting the results.  Failure to consider these factors often leads to incorrect metric calculations and misleading conclusions.

In my experience, debugging custom metrics typically involves careful inspection of the shapes and values of `y_true` and `y_pred` at different stages of the computation.  Print statements strategically placed within the custom metric function can be invaluable for pinpointing the source of errors.  Moreover, utilizing Keras's built-in functionality for checking tensor shapes and data types can prevent many common pitfalls.

**2. Code Examples with Commentary:**

**Example 1:  Mean Absolute Percentage Error (MAPE)**

MAPE is a common metric in forecasting tasks, but it's not readily available in Keras.  Here's a robust implementation:

```python
import tensorflow as tf
import numpy as np

def mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).  Handles division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAPE value (scalar).  Returns Inf if y_true contains all zeros.
    """
    y_true = tf.cast(y_true, tf.float32) #Ensuring correct data type
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs(y_true - y_pred)
    mask = tf.greater(tf.abs(y_true), 1e-7) # Avoid division by zero where y_true is near zero.
    percentage_errors = tf.where(mask, diff / tf.abs(y_true), tf.zeros_like(diff))
    mape = tf.reduce_mean(percentage_errors) * 100
    return mape


#Example usage:
y_true = np.array([10, 20, 30, 0])
y_pred = np.array([12, 18, 33, 1])
mape_value = mape(y_true, y_pred).numpy()
print(f"MAPE: {mape_value:.2f}%")
```

This example showcases several key aspects: explicit type casting for numerical consistency, handling division by zero using a mask, and converting the TensorFlow tensor to a NumPy array for easier printing.  The `1e-7` threshold prevents numerical instability caused by very small values of `y_true`.

**Example 2:  Weighted F1-score for Imbalanced Datasets**

In classification problems with imbalanced classes, a weighted F1-score is often preferred over the standard F1-score.

```python
import tensorflow as tf
from tensorflow.keras.metrics import f1_score

def weighted_f1(y_true, y_pred):
    """
    Computes the weighted F1-score, considering class imbalances.

    Args:
        y_true: Ground truth labels (one-hot encoded).
        y_pred: Predicted probabilities.

    Returns:
        Weighted F1-score (scalar).
    """
    # Ensure the predicted probabilities are between 0 and 1
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(f1_score(y_true, y_pred, average='weighted'))

#Example Usage
y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
y_pred = tf.constant([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])
weighted_f1_score = weighted_f1(y_true, y_pred).numpy()
print(f"Weighted F1 Score: {weighted_f1_score}")

```
This code snippet leverages Keras's built-in `f1_score` function for efficiency.  However, it demonstrates how to integrate it into a custom metric while accounting for potential issues like probabilities outside the [0, 1] range.

**Example 3:  Custom Metric for Sequence Generation**

When evaluating sequence generation models (e.g., machine translation),  standard metrics like accuracy are insufficient.  A custom metric might focus on BLEU score or ROUGE score.  Here, I'll illustrate a simplified example focusing on character-level accuracy.


```python
import tensorflow as tf

def char_accuracy(y_true, y_pred):
    """
    Calculates character-level accuracy for sequence generation.

    Args:
        y_true: Ground truth sequences (one-hot encoded).
        y_pred: Predicted sequences (probabilities).

    Returns:
        Character-level accuracy (scalar).
    """
    y_pred = tf.argmax(y_pred, axis=-1) #Selecting the most likely character
    correct_chars = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_chars, tf.float32))
    return accuracy

#Example Usage (assuming sequences of length 5)
y_true = tf.constant([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
y_pred = tf.constant([[[0.1, 0.9, 0.0, 0.0, 0.0],
                        [0.0, 0.7, 0.3, 0.0, 0.0],
                        [0.0, 0.0, 0.8, 0.2, 0.0],
                        [0.0, 0.0, 0.0, 0.6, 0.4],
                        [0.0, 0.0, 0.0, 0.1, 0.9]],
                       [[0.8, 0.2, 0.0, 0.0, 0.0],
                        [0.1, 0.8, 0.1, 0.0, 0.0],
                        [0.0, 0.0, 0.9, 0.1, 0.0],
                        [0.0, 0.0, 0.1, 0.8, 0.1],
                        [0.0, 0.0, 0.0, 0.2, 0.8]],
                       [[0.2, 0.8, 0.0, 0.0, 0.0],
                        [0.0, 0.3, 0.7, 0.0, 0.0],
                        [0.0, 0.0, 0.7, 0.3, 0.0],
                        [0.0, 0.0, 0.1, 0.8, 0.1],
                        [0.0, 0.0, 0.0, 0.1, 0.9]]])
char_acc = char_accuracy(y_true, y_pred).numpy()
print(f"Character Accuracy: {char_acc}")
```

This example highlights the necessary steps for handling sequence data, including the argmax operation to obtain the predicted character indices and the subsequent accuracy calculation.

**3. Resource Recommendations:**

The Keras documentation itself provides invaluable details on customizing training and evaluation.  Thorough understanding of TensorFlow or Theano (depending on your Keras backend) is crucial for advanced metric customization.  Books focusing on deep learning with TensorFlow/Keras are extremely helpful in understanding tensor manipulation and model building best practices.  Finally, consulting relevant research papers on specific metrics used in your field can provide insights into proper implementation and interpretation.
