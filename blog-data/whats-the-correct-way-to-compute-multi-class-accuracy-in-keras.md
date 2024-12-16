---
title: "What's the correct way to compute multi-class accuracy in Keras?"
date: "2024-12-16"
id: "whats-the-correct-way-to-compute-multi-class-accuracy-in-keras"
---

Alright, let's talk multi-class accuracy in Keras. It might seem straightforward at first glance, but there are nuances that can trip you up if you’re not careful. I’ve personally debugged enough model training sessions to know how crucial it is to get this metric right, especially when we're dealing with more than just binary classifications.

First off, understand that “accuracy” in this context refers to the percentage of predictions that exactly match the true labels. In the multi-class setting, this means every single prediction must align perfectly with the corresponding one-hot encoded true label. It’s not about being ‘close’, it’s about hitting the bullseye. Keras, thankfully, provides built-in tools to handle this effectively, but you need to configure them correctly based on your model's output and the format of your true labels. The most common stumbling block arises when you have your true labels in one form, like integer labels, and your model's output in another, say, probabilities or logits.

Let's break it down with a focus on practical implementations. Primarily, Keras’s `tf.keras.metrics.CategoricalAccuracy` is your go-to metric for most multi-class classification problems. However, how you use it depends heavily on your data and model setup. This metric requires that both the predictions and the true labels are in categorical format - specifically, they need to be one-hot encoded. It's crucial to keep the distinction between predictions and true labels, or you'll end up computing an entirely misleading accuracy, one that's probably too good to be true.

Now, suppose you’ve got your labels as integers representing class indices, like [0, 2, 1, 0, 3], and your model outputs probabilities for each class – essentially, a set of probabilities for each input, summing up to one. Here's how you handle this common scenario. We’ll use the `sparse_categorical_crossentropy` loss function, which expects integer encoded targets.

```python
import tensorflow as tf
import numpy as np

# Example data: integer labels and probability outputs
y_true_int = np.array([0, 2, 1, 0, 3])
y_pred_probs = np.array([
    [0.8, 0.1, 0.05, 0.05], # Prediction for label 0
    [0.1, 0.2, 0.6, 0.1], # Prediction for label 2
    [0.1, 0.7, 0.1, 0.1], # Prediction for label 1
    [0.7, 0.1, 0.1, 0.1], # Prediction for label 0
    [0.2, 0.1, 0.1, 0.6] # Prediction for label 3
])

# We convert integer labels to categorical using one-hot encoding.
y_true_cat = tf.keras.utils.to_categorical(y_true_int)

# Calculate accuracy manually to ensure we understand the computation.
y_pred_classes = np.argmax(y_pred_probs, axis=1)
manual_accuracy = np.mean(y_pred_classes == y_true_int)
print(f"Manual accuracy: {manual_accuracy}")

# Using CategoricalAccuracy - needs probabilities and one-hot encoded true labels.
metric = tf.keras.metrics.CategoricalAccuracy()
metric.update_state(y_true_cat, y_pred_probs)
keras_accuracy = metric.result().numpy()
print(f"Keras CategoricalAccuracy: {keras_accuracy}")

# SparseCategoricalAccuracy - Needs probabilities and integer labels.
sparse_metric = tf.keras.metrics.SparseCategoricalAccuracy()
sparse_metric.update_state(y_true_int, y_pred_probs)
keras_sparse_accuracy = sparse_metric.result().numpy()
print(f"Keras SparseCategoricalAccuracy: {keras_sparse_accuracy}")

```

This first code example directly demonstrates both a manual accuracy calculation as well as how to correctly apply `CategoricalAccuracy`. It highlights the one-hot encoding step. You'll notice it’s a direct match of the manual calculation. Additionally, it shows the use of `SparseCategoricalAccuracy` which directly uses integer labels.

Now, what if your model outputs logits, which are raw, unnormalized scores, instead of probabilities? Typically, you'd run logits through a softmax activation function to get probabilities. Keras’s metric can handle this indirectly as you can pass the logits directly. This is convenient, but also introduces a common mistake that I have made in the past. If you have logits and pass to `CategoricalAccuracy`, it will interpret these as probabilities. Let’s do a second example with logits. Here the `SparseCategoricalAccuracy` comes into play.

```python
import tensorflow as tf
import numpy as np

# Example data: integer labels and logit outputs
y_true_int = np.array([0, 2, 1, 0, 3])
y_pred_logits = np.array([
    [2.1, -1.2, -0.5, -0.4],  # Logits for label 0
    [-1.1, -0.5, 2.5, -0.1],  # Logits for label 2
    [-0.2, 3.1, -0.3, -0.8],  # Logits for label 1
    [1.7, -0.4, -0.5, -0.8],  # Logits for label 0
    [-0.8, -0.1, -0.2, 2.3]   # Logits for label 3
])

# SparseCategoricalAccuracy - Needs logits and integer labels.
sparse_metric_logits = tf.keras.metrics.SparseCategoricalAccuracy()
sparse_metric_logits.update_state(y_true_int, y_pred_logits)
keras_sparse_accuracy_logits = sparse_metric_logits.result().numpy()
print(f"Keras SparseCategoricalAccuracy (logits): {keras_sparse_accuracy_logits}")

# Trying CategoricalAccuracy with logits - a common pitfall.
metric_logits = tf.keras.metrics.CategoricalAccuracy()
y_true_cat = tf.keras.utils.to_categorical(y_true_int)
metric_logits.update_state(y_true_cat, y_pred_logits)
keras_accuracy_logits = metric_logits.result().numpy()
print(f"Keras CategoricalAccuracy (logits - incorrect use): {keras_accuracy_logits}")

```
This second snippet emphasizes the correct use of `SparseCategoricalAccuracy` with logits, and shows what happens when logits are used incorrectly with `CategoricalAccuracy`. The results differ significantly, underscoring the critical importance of matching the metric and input types. You must correctly use `CategoricalAccuracy` only when you have one-hot encoded labels and probability outputs, not logits.

Finally, let’s illustrate a case with a model built using keras and incorporating the metric when the model is compiled.

```python
import tensorflow as tf
import numpy as np

# Generate a synthetic dataset
num_samples = 1000
num_classes = 5
input_shape = (10,)  # Example input shape

X = np.random.rand(num_samples, *input_shape)
y = np.random.randint(0, num_classes, num_samples)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax
])

# Compile the model using SparseCategoricalAccuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #integer targets
              metrics=['sparse_categorical_accuracy']) #integer targets and prob outputs

# Train the model (using dummy data)
model.fit(X, y, epochs=10, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Model accuracy: {accuracy}")
```

Here, the model is configured using `sparse_categorical_crossentropy` to allow for integer labels, and the `sparse_categorical_accuracy` metric to accurately compute the metric with integer labels and output probabilities.

In conclusion, choosing the right accuracy metric in Keras requires careful consideration of your output format. Remember, `CategoricalAccuracy` needs one-hot encoded true labels and probability outputs, while `SparseCategoricalAccuracy` expects integer true labels and either probability or logit outputs. Using the wrong metric can lead to severely misleading results. Always double-check the expected format of both true labels and model outputs.

For further study, I’d recommend exploring the Keras API documentation directly, specifically around `tf.keras.metrics` and the different loss functions like `tf.keras.losses.sparse_categorical_crossentropy`. Textbooks on deep learning will provide a thorough theoretical foundation. The book "Deep Learning with Python" by Francois Chollet will be a useful resource for how these metrics and loss functions are practically used in keras. Another highly regarded resource is "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron, which provides a detailed understanding of the mathematics involved, including how these metrics are calculated. Finally, always refer back to TensorFlow's own documentation which is the most authorative. It’s a habit that will save you a lot of frustration in the long run.
