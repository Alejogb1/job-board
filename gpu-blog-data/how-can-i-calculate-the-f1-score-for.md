---
title: "How can I calculate the F1 score for a Roberta model with one-hot encoded outputs?"
date: "2025-01-30"
id: "how-can-i-calculate-the-f1-score-for"
---
The crucial aspect to understand when calculating the F1 score for a RoBERTa model with one-hot encoded outputs lies in the inherent mismatch between the model's probabilistic predictions and the discrete nature of the one-hot encoding.  RoBERTa, like many transformer models, outputs probabilities across classes.  Directly comparing these probabilities to a one-hot vector necessitates careful handling to avoid inaccuracies.  In my experience developing and evaluating sentiment analysis systems using RoBERTa, I've found that neglecting this nuance often leads to misleading F1 scores.

**1. Clear Explanation:**

The F1 score is the harmonic mean of precision and recall, offering a balanced measure of a classifier's performance.  It's particularly valuable when dealing with imbalanced datasets.  For a multi-class classification problem, like one using a one-hot encoded output representing multiple sentiments (positive, negative, neutral), a micro-averaged or macro-averaged F1 score is typically employed.

The micro-averaged F1 score aggregates the true positives, false positives, and false negatives across all classes before calculating precision and recall. This approach is less sensitive to class imbalance than the macro-averaged F1 score, which calculates the F1 score for each class individually and then averages them.  The choice between micro and macro averaging depends on the specific application and the desired emphasis on individual class performance versus overall performance.

With one-hot encoded outputs, the RoBERTa model needs to provide a probability distribution over the classes.  We then determine the predicted class by selecting the class with the highest probability.  This predicted class is then compared to the true class represented by the one-hot vector. From these comparisons, we compute the true positives, false positives, and false negatives necessary for calculating the F1 score.  Let's break down the process step-by-step:

1. **Obtain RoBERTa model predictions:** The model will output a probability vector for each input sample.  For example, if you have three classes (positive, negative, neutral), the output might look like `[0.7, 0.2, 0.1]`.

2. **Convert probabilistic predictions to class labels:** Find the index of the maximum probability in the prediction vector.  This index corresponds to the predicted class.  In the example above, the predicted class would be 'positive' (index 0).

3. **Compare predictions to one-hot encoded true labels:**  Compare the predicted class to the true class from the one-hot encoded ground truth.  For instance, if the true label is `[1, 0, 0]` (positive), it's a true positive.  If the true label is `[0, 1, 0]` (negative), it's a false negative from the perspective of the 'positive' class and a false positive from the perspective of the 'negative' class.

4. **Calculate precision and recall for each class:**  Calculate precision (true positives / (true positives + false positives)) and recall (true positives / (true positives + false negatives)) for each class separately.

5. **Calculate micro-averaged or macro-averaged F1 score:**  Use the calculated precision and recall values to determine the micro-averaged or macro-averaged F1 score as appropriate.  The formula for the F1 score is: 2 * (precision * recall) / (precision + recall).


**2. Code Examples with Commentary:**

**Example 1: Using Scikit-learn (Micro-Averaged F1)**

```python
import numpy as np
from sklearn.metrics import f1_score

# Predicted probabilities from RoBERTa (example for 3 samples and 3 classes)
y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.1, 0.6]])

# Convert probabilities to class labels
y_pred = np.argmax(y_prob, axis=1)

# One-hot encoded true labels (example)
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Calculate micro-averaged F1 score
f1_micro = f1_score(y_true.argmax(axis=1), y_pred, average='micro')
print(f"Micro-averaged F1 score: {f1_micro}")
```

This example demonstrates a straightforward calculation of the micro-averaged F1 score using scikit-learn's `f1_score` function.  The `argmax` function efficiently converts both the predicted probabilities and the one-hot encoded labels to their respective class indices.

**Example 2: Manual Calculation (Macro-Averaged F1)**

```python
import numpy as np

# Predicted probabilities and one-hot encoded true labels (same as Example 1)
y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.1, 0.6]])
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.argmax(y_prob, axis=1)

num_classes = y_true.shape[1]
f1_scores = []

for i in range(num_classes):
    tp = np.sum((y_pred == i) & (y_true[:, i] == 1))
    fp = np.sum((y_pred == i) & (y_true[:, i] == 0))
    fn = np.sum((y_pred != i) & (y_true[:, i] == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

f1_macro = np.mean(f1_scores)
print(f"Macro-averaged F1 score: {f1_macro}")
```

This example shows a manual calculation of the macro-averaged F1 score, illustrating the step-by-step process described in the explanation.  It explicitly handles potential division by zero errors.

**Example 3: Using TensorFlow/Keras (Micro-Averaged F1)**

```python
import tensorflow as tf
import numpy as np

# Assuming 'model' is your trained RoBERTa model
# y_true is a NumPy array of one-hot encoded labels
# X_test is your test data

y_prob = model.predict(X_test)
y_pred = tf.argmax(y_prob, axis=1).numpy()
y_true = tf.argmax(y_true, axis=1).numpy()


f1 = tf.keras.metrics.F1Score(num_classes=3, average='micro')
f1.update_state(y_true, y_pred)
micro_f1 = f1.result().numpy()

print(f"Micro-averaged F1 score: {micro_f1}")
```

This example demonstrates the integration with TensorFlow/Keras for calculating the micro-averaged F1 score.  It leverages Keras's built-in `F1Score` metric for efficiency and consistency with the rest of the TensorFlow workflow.  Remember to adjust the `num_classes` argument according to your specific task.


**3. Resource Recommendations:**

Scikit-learn documentation, TensorFlow documentation, dedicated textbooks on machine learning evaluation metrics.  A thorough understanding of probability theory and statistical inference is beneficial.  Focusing on practical applications through various projects further solidifies the understanding of these concepts.
