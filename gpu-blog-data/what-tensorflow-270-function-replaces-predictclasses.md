---
title: "What TensorFlow 2.7.0 function replaces `predict_classes()`?"
date: "2025-01-30"
id: "what-tensorflow-270-function-replaces-predictclasses"
---
The shift from TensorFlow 1.x to 2.x involved significant API changes, notably the removal of several convenience functions. Specifically, the `predict_classes()` method, commonly used in TensorFlow 1.x for classification model predictions, was deprecated and is not present in TensorFlow 2.7.0. I encountered this directly while migrating a legacy image classification pipeline from TF 1.15 to TF 2.8 last year. The core issue stems from TensorFlow 2's emphasis on explicit computation graphs and Eager Execution, which required a more flexible approach to extracting predictions. The replacement methodology involves applying a model's output to obtain the class indices directly using `tf.argmax()`.

In TensorFlow 1.x, `predict_classes()` implicitly performed two steps: it first generated the raw output from the model, which often represents logits or probabilities for each class; then, it extracted the index of the class with the highest probability. This streamlined the prediction process, but it masked the underlying operations. The TensorFlow 2 approach offers more control and transparency. Now, the model's prediction output needs to be processed to get class predictions.  For models that output logits (raw, unscaled scores), one might first apply `tf.nn.softmax()` to convert the logits into probabilities before identifying the class with the highest probability. However, many classification models output probabilities directly as a result of a softmax activation layer being included at the very end of the model's structure. Thus, it is most common to utilize `tf.argmax()` directly on the output, since we are not interested in knowing the class probabilities, but instead the *predicted* class.

Here's a more detailed breakdown with illustrative code examples:

**Example 1: Basic Classification with Logits Output**

Imagine we have a simple classification model that outputs logits (raw scores) for a batch of data. I've often seen models implemented like this, especially in older codebases, where a softmax activation is sometimes omitted from the very end.

```python
import tensorflow as tf
import numpy as np

# Assume 'model' is a pre-trained TensorFlow model.
# For demonstration, we will use a dummy model: a dense layer.

model = tf.keras.layers.Dense(units=10) # 10 classes
dummy_input = tf.random.normal(shape=(32, 128)) # Batch size 32, feature size 128

logits = model(dummy_input)
# logits now contains raw, unscaled output scores
# each row is output from one input sample and each column is the score of each class
# example: logits.shape == (32, 10)

# Before the prediction, one should first convert them to probabilities
probabilities = tf.nn.softmax(logits, axis=1) # Softmax on axis 1
predicted_classes = tf.argmax(probabilities, axis=1)

print("Logits shape:", logits.shape)
print("Probabilities shape:", probabilities.shape)
print("Predicted class indices:", predicted_classes.numpy()) # Predicted class indices, size: (32,)
```

**Commentary:** In this example, I first create a dummy model that outputs logits (raw scores for each class).  The `model(dummy_input)` operation generates these logits.  Then, `tf.nn.softmax()` transforms these logits into class probabilities. Finally,  `tf.argmax(probabilities, axis=1)` selects the class index with the highest probability along the *class* axis (axis=1). The `axis=1` parameter specifies that the maximum should be computed per sample, not for the whole batch. The resulting `predicted_classes` tensor contains the class index predicted by the model for each input sample in the batch. This directly mirrors the function of `predict_classes()`, but allows intermediate steps to be inspected or altered if needed. This is how I most frequently extract class predictions after training and evaluation.

**Example 2: Model Directly Outputting Probabilities**

Often, I see models where the final layer *does* apply the softmax activation such that the model outputs probabilities.  In this case, you skip the additional step of applying `tf.nn.softmax()` and apply `tf.argmax()` directly.

```python
import tensorflow as tf
import numpy as np

# Example model with softmax in the last layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='softmax') # 10 classes, softmax activation
])

dummy_input = tf.random.normal(shape=(32, 128)) # Batch size 32, feature size 128
probabilities = model(dummy_input) # Output probabilities directly

predicted_classes = tf.argmax(probabilities, axis=1)

print("Probabilities shape:", probabilities.shape)
print("Predicted class indices:", predicted_classes.numpy()) # Predicted class indices, size: (32,)
```

**Commentary:** This example demonstrates a model that incorporates a softmax activation in its final layer. Consequently, `model(dummy_input)` directly returns class probabilities. This is the format I tend to prefer as it is directly interpretable. Thus,  `tf.argmax()` directly selects the index with the highest probability. The structure is simpler and avoids an unnecessary step, which I often aim for when I can guarantee that my model output is in the format I expect.  Again, `axis=1` ensures the maximum probability per sample is used for index selection.

**Example 3: Handling Multi-Label Classification**

While `predict_classes()` was designed for single-label classification, `tf.argmax()` can be adapted for multi-label scenarios (although not directly).  Here, a more nuanced approach is required. While technically `predict_classes()` does not exist in TensorFlow 2.7, we can use `tf.sigmoid` or `tf.round()` with `tf.cast()`. This is the least common, but still very necessary in some multi-label classification tasks I have encountered. Here I will demonstrate sigmoid, in which each output node is independent, allowing for multiple positive labels.

```python
import tensorflow as tf
import numpy as np

# Example model with sigmoid output for multi-label classification
model = tf.keras.layers.Dense(units=5, activation='sigmoid') # 5 labels, sigmoid activation
dummy_input = tf.random.normal(shape=(10, 64)) # Batch size 10, feature size 64
logits = model(dummy_input) # Output logits (unscaled scores)

# Apply sigmoid activation for independent probabilities
probabilities = tf.sigmoid(logits)

# Apply threshold for binary predictions. 0.5 is the standard threshold.
# This can be changed for more precise classifications based on the application
threshold = 0.5
binary_predictions = tf.cast(tf.greater(probabilities, threshold), tf.int32)

print("Probabilities shape:", probabilities.shape)
print("Binary predictions shape:", binary_predictions.shape)
print("Binary predictions:", binary_predictions.numpy())
```

**Commentary:** In multi-label classification, a single input can have multiple associated classes. Instead of `softmax()`, which forces a single classification output, the activation function in this case is `sigmoid()`, which can return a probability for each class, independently. The user needs to choose a threshold to convert these probabilities to binary predictions (1 for the class being "present," 0 for the class being absent). Using a threshold like 0.5 is the simplest strategy, but in practice, a user may find a more optimal choice through data analysis or other methods. In this approach, I use `tf.sigmoid` and `tf.greater` to threshold these individual class predictions and `tf.cast` to convert to integer labels. Here we obtain a tensor of 1s and 0s indicating the presence or absence of each class for each input. This solution represents my general approach for binary multi-label problems, or one that I most frequently encounter.

In summary, `tf.argmax()` is the core mechanism replacing `predict_classes()`. It directly extracts the predicted class indices, offering flexibility. It may or may not require preprocessing with `tf.nn.softmax()` before the class indices are obtained. For multi-label classifications, `tf.sigmoid()` or `tf.round()` with `tf.cast()` can be employed, each offering varying properties for specific needs. This approach aligns with TensorFlow 2’s design principles, fostering explicit understanding and greater user control.

**Recommended Resources:**

For a deeper understanding of the TensorFlow API and the migration process, I recommend these resources:

1. **The official TensorFlow documentation:** This provides comprehensive explanations of functions and their usage, including `tf.argmax()` and `tf.nn.softmax()`. The examples provided on the official API pages are very useful.
2. **TensorFlow tutorials:** These offer practical examples on building and using models, covering classification tasks and specific use cases of `tf.argmax()`.  The official examples are generally quite robust and up-to-date, making them easy to implement and adapt.
3. **TensorFlow guides:** These guides explain the design principles of TensorFlow 2, which are extremely useful for understanding *why* changes were made and how to effectively apply new methods. The guides tend to dive into the theory more often than the tutorials.
4. **Books on deep learning with TensorFlow:** Books can provide a more structured and comprehensive understanding of deep learning concepts and their implementation in TensorFlow.
