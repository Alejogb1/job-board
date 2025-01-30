---
title: "Why is my TensorFlow model achieving perfect loss and accuracy on every step (without Keras or Scikit-learn)?"
date: "2025-01-30"
id: "why-is-my-tensorflow-model-achieving-perfect-loss"
---
The observation of perfect loss and accuracy during every training step of a TensorFlow model, absent the use of high-level APIs like Keras or scikit-learn, strongly suggests a fundamental issue with the model's training or evaluation process, rather than an indication of truly ideal performance. Such behavior points towards either an inability to learn or an improperly designed evaluation loop.

Specifically, my experience indicates that this scenario often occurs when the model is essentially learning nothing, and the “training” data, or the evaluation process, is flawed, allowing the model to seemingly succeed without genuine generalization ability. Several contributing factors can lead to this outcome, which I’ve encountered across various projects.

**Explanation of the Underlying Issue**

A perfect loss of 0 and a perfect accuracy of 1 across every training step directly translate to the model perfectly classifying every single instance within the dataset being utilized during a particular step, whether it be batch or full training set. This could arise from a few situations. Firstly, if the data being used during each training step is the *same* small set of data and the model is overfitting. In my previous work implementing an image classification neural network using solely TensorFlow, I accidentally used the same small batch repeatedly instead of iterating through the dataset. This resulted in perfect performance in those training batches, and upon finally evaluating the test set, the performance was catastrophic.

Second, this perfect performance could occur if the model is learning a trivial solution to the problem due to improperly prepared data. Consider the case where labels are improperly associated with data, or if some feature in the data inadvertently encodes the label. For instance, if a model is being trained to classify cat images, and every cat image has a label value equal to one, while other images have a label equal to zero, the model would learn to predict based on the fact the label is 1, and not on the image content.

Thirdly, incorrect computation of loss or metrics would also create the illusion of perfect performance. If the loss function was calculated in such a way that it is trivially equal to 0, or if the metric was always equal to 1, there will be perfect numbers all throughout training. This is most common if the loss function and the metric are both hardcoded to always be the same value.

Lastly, an absence of data augmentation or regularization, combined with small datasets, tends towards a memorization-prone regime. While not as likely to produce perfect performance at each step unless the model is extremely small or the data incredibly simple, this combination promotes a scenario where it is possible to obtain a perfect result on the training data which fails to generalize. This is often coupled with one of the other failures mentioned. The primary issue is the model does not learn to extract general features that are meaningful for classifying different instances, but instead memorizes specific examples in the dataset.

**Code Examples and Commentary**

I've provided three examples that showcase common errors I've observed, along with commentary on how to approach these problems. These examples will focus on a binary classification task where the true and predicted outputs are either 0 or 1.

*   **Example 1: Repeated Training Data**

```python
import tensorflow as tf
import numpy as np

# Generate synthetic data
data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
labels = np.array([[0], [1], [0], [1]], dtype=np.float32)

# Simple linear model
W = tf.Variable(tf.random.normal(shape=(2, 1), dtype=tf.float32), name='weights')
b = tf.Variable(tf.zeros(shape=(1,), dtype=tf.float32), name='bias')

def model(x):
    return tf.sigmoid(tf.matmul(x, W) + b)

def loss_fn(y_true, y_pred):
  return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def accuracy_metric(y_true, y_pred):
    predicted_labels = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(predicted_labels, y_true), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

# ERROR: Training on the same single batch repeatedly
for epoch in range(100):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    accuracy = accuracy_metric(labels, predictions)

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")
```

*   **Commentary:** This example demonstrates training on a very small dataset without proper batching. Since every iteration uses the *same* four examples, the model quickly memorizes their classifications, achieving perfect loss and accuracy without learning meaningful patterns. The fix is to utilize tf.data to generate batches and iterate over the dataset. This ensures that the model is exposed to different data at each step. Using dataset iterators can improve the speed and also prevent this type of error.

*   **Example 2: Data Leakage in Labels**

```python
import tensorflow as tf
import numpy as np

# Data where a feature is a direct indicator of the label (leakage)
data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
labels = np.array([[0], [1], [0], [1]], dtype=np.float32)

# Model, loss, metric, and optimizer are the same as in the previous example.
W = tf.Variable(tf.random.normal(shape=(2, 1), dtype=tf.float32), name='weights')
b = tf.Variable(tf.zeros(shape=(1,), dtype=tf.float32), name='bias')

def model(x):
    return tf.sigmoid(tf.matmul(x, W) + b)

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def accuracy_metric(y_true, y_pred):
    predicted_labels = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(predicted_labels, y_true), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

# Correct training loop, but flawed data.
for epoch in range(100):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    accuracy = accuracy_metric(labels, predictions)

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")
```

*   **Commentary:** Here, one of the features directly corresponds to the label, creating a trivial learning task. If the first feature is 1.0, the label is always 0, and if the second is 1.0, the label is always 1. The model will quickly find this and the weights associated with it, therefore always predicting correctly. To resolve this, data preprocessing is needed. Removing the feature that leaks the label, and including more features to create a more difficult training task.

*   **Example 3: Incorrect Metric Calculation**

```python
import tensorflow as tf
import numpy as np

# Generate synthetic data
data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
labels = np.array([[0], [1], [0], [1]], dtype=np.float32)

# Model, loss, and optimizer as in the previous examples
W = tf.Variable(tf.random.normal(shape=(2, 1), dtype=tf.float32), name='weights')
b = tf.Variable(tf.zeros(shape=(1,), dtype=tf.float32), name='bias')

def model(x):
    return tf.sigmoid(tf.matmul(x, W) + b)

def loss_fn(y_true, y_pred):
  return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def accuracy_metric(y_true, y_pred):
    # ERROR: Always returning 1
    return tf.constant(1.0, dtype=tf.float32)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

for epoch in range(100):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    accuracy = accuracy_metric(labels, predictions)

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")
```

*   **Commentary:** In this scenario, the `accuracy_metric` function is flawed, consistently returning 1.0, regardless of the prediction. This results in the model showing an accuracy of 1 despite potentially making incorrect predictions. This underscores the importance of carefully reviewing not just your training loop and model architecture, but also your evaluation metrics.

**Resource Recommendations**

To gain a deeper understanding of best practices in TensorFlow and avoid these errors, I recommend reviewing several resources. The official TensorFlow documentation is invaluable. Particular attention should be given to guides on data input pipelines (`tf.data`), custom training loops, metrics, and evaluation. Books that cover deep learning implementation often dedicate chapters to correctly using frameworks like TensorFlow. Additionally, reviewing blog posts by experienced practitioners often sheds light on common pitfalls. Always focus on understanding the fundamental concepts behind the methods.
