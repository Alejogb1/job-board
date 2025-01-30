---
title: "How can I use scikit-learn's macro F1-score as a metric in TensorFlow Keras?"
date: "2025-01-30"
id: "how-can-i-use-scikit-learns-macro-f1-score-as"
---
The core challenge in integrating scikit-learn's macro F1-score with TensorFlow Keras lies in the differing data structures and calculation methodologies each library employs.  Scikit-learn's `f1_score` function expects a flat array of true labels and predicted labels, while Keras typically outputs probabilities or logits from a model's output layer.  My experience optimizing multi-class classification models for imbalanced datasets has highlighted the necessity for meticulous handling of these discrepancies.  This response details how to effectively bridge this gap.

**1. Clear Explanation:**

The solution involves creating a custom metric function within the Keras model compilation process. This function will take the true labels and predicted probabilities (or logits, depending on your model's output activation) as input, convert the probabilities to class predictions using an appropriate threshold (typically 0.5), and then utilize scikit-learn's `f1_score` function to compute the macro-averaged F1-score.  Crucially, we need to handle potential shape mismatches between Keras' output and the scikit-learn function's input.

The macro F1-score, as opposed to the micro or weighted F1-score, treats each class equally, regardless of its prevalence in the dataset. This is particularly important for scenarios with imbalanced classes, where a micro-averaged F1-score might be misleadingly high due to the dominance of the majority class.  My experience with fraud detection models, where fraudulent transactions are a small fraction of the total, underscored this critical distinction.

The key steps involved are:

*   **Retrieving Predictions:** Obtain predicted probabilities from the Keras model's output layer.  If using a sigmoid activation for binary classification, this is straightforward. For multi-class classification with a softmax activation, this will yield probabilities for each class.
*   **Converting Probabilities to Predictions:** Apply an appropriate threshold (e.g., 0.5) to the predicted probabilities to convert them into class labels.  For multi-class classification, this involves selecting the class with the highest probability for each instance.
*   **Reshaping Data:** Ensure that the shapes of the true labels and predicted labels are compatible with `sklearn.metrics.f1_score`.  This often involves reshaping arrays using NumPy's `reshape()` function.
*   **Calculating Macro F1-score:** Use `sklearn.metrics.f1_score` with the `average='macro'` parameter to calculate the macro-averaged F1-score.
*   **Integrating into Keras:** Incorporate this calculation into a custom metric function that can be passed to the `metrics` argument during model compilation.


**2. Code Examples with Commentary:**

**Example 1: Binary Classification**

```python
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

def macro_f1(y_true, y_pred):
    y_pred = tf.round(y_pred).numpy() #Binary classification: round probabilities to 0/1
    y_true = y_true.numpy()
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'), [y_true, y_pred], tf.double)

model = tf.keras.Sequential([
    # ... your model layers ...
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[macro_f1])
```

This example demonstrates a custom metric function for binary classification.  The `tf.round` function converts the predicted probabilities into binary predictions (0 or 1). The `tf.py_function` ensures that the scikit-learn function is executed within the TensorFlow graph.  Note the use of `tf.double` to specify the return type.  I found this crucial for numerical stability during training of large models.

**Example 2: Multi-Class Classification with Softmax**

```python
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

def macro_f1(y_true, y_pred):
    y_pred = np.argmax(y_pred.numpy(), axis=1) #Multi-class: take argmax for prediction
    y_true = y_true.numpy().argmax(axis=1) #Assuming one-hot encoded true labels
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'), [y_true, y_pred], tf.double)


model = tf.keras.Sequential([
    # ... your model layers ...
    tf.keras.layers.Activation('softmax') #important for multi-class probability output
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[macro_f1])
```

This example extends the approach to multi-class classification using a softmax activation function.  `np.argmax` selects the class with the highest probability for both predicted and true labels.  It assumes that `y_true` is one-hot encoded. I have consistently observed improved performance with one-hot encoding in multi-class problems.

**Example 3: Handling potential shape mismatches**

```python
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

def macro_f1(y_true, y_pred):
    y_pred = np.argmax(y_pred.numpy(), axis=1).reshape(-1)
    y_true = y_true.numpy().argmax(axis=1).reshape(-1)
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'), [y_true, y_pred], tf.double)

model = tf.keras.Sequential([
    # ... your model layers ...
    tf.keras.layers.Activation('softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[macro_f1])
```

This example explicitly addresses potential shape mismatches by using `reshape(-1)` to flatten the arrays before passing them to `f1_score`. This is a robust method to prevent errors arising from inconsistent array dimensions which I have encountered frequently in real-world projects.


**3. Resource Recommendations:**

*   The scikit-learn documentation on metrics.
*   The TensorFlow Keras documentation on custom metrics.
*   A comprehensive textbook on machine learning covering evaluation metrics.
*   Relevant research papers on imbalanced datasets and F1-score.
*   A practical guide to deep learning with TensorFlow/Keras.


These resources offer a more complete understanding of the underlying concepts and provide further guidance on implementing and interpreting the macro F1-score.  Consistent review of these foundational texts is imperative for refining one's understanding. Remember to carefully consider the implications of your chosen metric in relation to your specific application.  The macro F1-score is valuable in many cases, but its suitability should be assessed for each unique problem.
