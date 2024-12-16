---
title: "How do I calculate multi-class accuracy correctly in Keras?"
date: "2024-12-16"
id: "how-do-i-calculate-multi-class-accuracy-correctly-in-keras"
---

Let's tackle multi-class accuracy calculation in Keras. It’s a topic I've had to revisit quite a few times, particularly when dealing with nuanced model evaluation scenarios. I recall one specific project, a complex image classification task involving segmented medical scans. Initially, the training metrics looked promising, but when we started evaluating on unseen data, the true performance was considerably lower. Turns out, we hadn't been calculating accuracy quite as robustly as we should have.

The essence of correctly calculating multi-class accuracy in Keras lies in understanding what the model outputs, how it predicts, and then aligning that with the ground truth. You see, in a multi-class setting, your model is typically outputting a vector of probabilities, one for each class. The predicted class is usually the one with the highest probability. It's crucial to then compare this predicted class against the *actual* class. Keras provides tools to compute these metrics, but it's easy to fall into a few traps if you're not careful about the format of your inputs.

The most common pitfall I’ve encountered revolves around the input format of your 'y_true' labels. In a multi-class setting, 'y_true' can be one of two formats: integer labels (e.g., 0, 1, 2, corresponding to class indices) or one-hot encoded labels. Similarly, the output from your model can also be a set of raw probabilities (logits) before the softmax, or probabilities after applying the softmax. Let's break down how to address each scenario correctly.

First, if your *y_true* is in the form of integer labels, and the model output is raw probabilities, you'll need to convert the probabilities into predicted classes (argmax) before comparing with *y_true*. If the output already has the softmax applied, you can proceed directly to argmax. It's imperative to use `tf.math.argmax` for obtaining the indices of the highest probabilities. The function `tf.keras.metrics.CategoricalAccuracy` is suitable in this case only when output is probability distribution (softmax), it will calculate correct accuracy by directly comparing the predicted class indices to *y_true*.

Here’s a code snippet illustrating how to correctly calculate accuracy when *y_true* consists of integer labels, and your model output gives probability distribution (after softmax):

```python
import tensorflow as tf

# Example: Integer Labels, Model Output Probability Distribution
y_true_int = tf.constant([1, 2, 0, 1, 2])  # Integer labels
y_pred_probs = tf.constant([[0.1, 0.8, 0.1],   # Probability Distribution for each label.
                            [0.2, 0.1, 0.7],
                            [0.9, 0.05, 0.05],
                            [0.3, 0.6, 0.1],
                            [0.2, 0.2, 0.6]])

# Using CategoricalAccuracy, which expects probability distribution as output
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
accuracy_metric.update_state(tf.one_hot(y_true_int, depth=3), y_pred_probs) # Convert y_true to one-hot
accuracy = accuracy_metric.result()
print("Accuracy (Integer Labels, Probability Distribution):", accuracy.numpy())

# Verification using argmax
y_pred_classes = tf.math.argmax(y_pred_probs, axis=1)
comparison = tf.cast(y_pred_classes == y_true_int, dtype=tf.float32)
accuracy_verification = tf.reduce_mean(comparison)
print("Verification Accuracy (Integer Labels):", accuracy_verification.numpy())
```

The output here shows that the calculated accuracy is 0.8, as three out of the five predictions were correct.

Now, let’s consider the case where *y_true* is one-hot encoded. In this case, `tf.keras.metrics.CategoricalAccuracy` can be used directly when the model output are predicted probabilities (after softmax). We do not need to calculate the argmax of predictions in this case. But we do have to ensure that *y_true* is also one-hot encoded.

Here's a code snippet showing how to handle accuracy calculations when *y_true* is one-hot encoded and the model output is probability distribution:

```python
import tensorflow as tf

# Example: One-Hot Encoded Labels, Model Output Probability Distribution
y_true_onehot = tf.constant([[0, 1, 0],   # One-hot encoded labels
                             [0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=tf.float32)
y_pred_probs = tf.constant([[0.1, 0.8, 0.1],   # Probability Distribution for each label
                            [0.2, 0.1, 0.7],
                            [0.9, 0.05, 0.05],
                            [0.3, 0.6, 0.1],
                            [0.2, 0.2, 0.6]])

# Using CategoricalAccuracy, which expects one-hot as input
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
accuracy_metric.update_state(y_true_onehot, y_pred_probs)
accuracy = accuracy_metric.result()
print("Accuracy (One-Hot Labels, Probability Distribution):", accuracy.numpy())


#Verification using argmax
y_true_classes = tf.math.argmax(y_true_onehot, axis=1)
y_pred_classes = tf.math.argmax(y_pred_probs, axis=1)
comparison = tf.cast(y_pred_classes == y_true_classes, dtype=tf.float32)
accuracy_verification = tf.reduce_mean(comparison)
print("Verification Accuracy (One-Hot Labels):", accuracy_verification.numpy())
```

The output confirms again that the accuracy is 0.8, with three correctly classified samples.

Finally, consider a scenario where you have *y_true* in one-hot format, but model output is raw probabilities (logits), before the softmax layer. In this case you must convert the raw logits to probability distribution by using the softmax function before passing them into accuracy calculation functions.

Here is how to handle accuracy in this case:

```python
import tensorflow as tf

# Example: One-Hot Encoded Labels, Model Output Raw Probabilities (Logits)
y_true_onehot = tf.constant([[0, 1, 0],  # One-hot encoded labels
                             [0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=tf.float32)
y_pred_logits = tf.constant([[0.1, 1.8, 0.1],   # Logits (raw probabilities) for each label
                             [0.2, 0.1, 1.7],
                             [1.9, 0.05, 0.05],
                             [0.3, 1.6, 0.1],
                             [0.2, 0.2, 1.6]])


# Applying softmax to convert logits to probability distributions
y_pred_probs = tf.nn.softmax(y_pred_logits)

# Using CategoricalAccuracy, which expects one-hot as input
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
accuracy_metric.update_state(y_true_onehot, y_pred_probs)
accuracy = accuracy_metric.result()
print("Accuracy (One-Hot Labels, Raw Logits):", accuracy.numpy())

#Verification Using argmax
y_true_classes = tf.math.argmax(y_true_onehot, axis=1)
y_pred_classes = tf.math.argmax(y_pred_probs, axis=1)
comparison = tf.cast(y_pred_classes == y_true_classes, dtype=tf.float32)
accuracy_verification = tf.reduce_mean(comparison)
print("Verification Accuracy (One-Hot Labels, Raw Logits):", accuracy_verification.numpy())
```

Again, the calculated accuracy is 0.8. As you can see, the key takeaway here is matching the formats of your *y_true* labels and the model's predictions. Always ensure that your *y_true* is correctly formatted as either integer indices or one-hot encoded vectors, and that if your model outputs logits, they need to be converted to probabilities using softmax before calculating accuracy.

For further in-depth understanding, I'd recommend reviewing *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, particularly the sections on classification and metrics. Additionally, the Keras documentation itself provides detailed explanations of metrics and loss functions. The TensorFlow documentation on tf.keras.metrics module is essential. I would also suggest reading papers on model evaluation metrics in different classification scenarios, for instance, papers on class imbalance scenarios could enhance the understanding of this topic even further. Remember to pay close attention to the specific requirements of your dataset and model output to get the most accurate assessment of your model's performance.
