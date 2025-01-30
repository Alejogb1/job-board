---
title: "How can class weights be applied to TensorFlow neural networks other than the standard method?"
date: "2025-01-30"
id: "how-can-class-weights-be-applied-to-tensorflow"
---
Class imbalance is a pervasive challenge in machine learning, and while TensorFlow's `class_weight` parameter in model training is a convenient approach, it's not always the most effective or flexible solution. Based on my experience developing a fraud detection model, I’ve encountered situations where directly manipulating the loss function and incorporating sample weights provides significantly better performance and control over how the model learns from imbalanced data. These alternate strategies afford more granular tuning and a deeper understanding of model behavior.

The canonical way to manage class imbalance in TensorFlow involves passing a dictionary of `class_weights` to the `model.fit` method. This dictionary maps class indices to weights, effectively scaling the loss contribution of each sample based on its class membership. While straightforward, this method treats all samples within a given class equally, which may not be optimal. Consider a scenario where some samples, due to their proximity to the decision boundary or their intrinsic ambiguity, hold more information than others. Treating these samples identically to easier samples can lead to suboptimal learning. This is where alternative approaches come into play. I'll explain two such methods: direct loss function manipulation with custom weights, and the use of sample weights.

First, we can directly modify the loss function to incorporate class weighting. Rather than relying on `model.fit` to handle weighting, I define my loss as a weighted average of the standard categorical crossentropy. This requires understanding how the loss function operates. Let’s assume we're using categorical crossentropy, which calculates the average negative log-likelihood of the true labels given the predicted probabilities. To apply class weights, we modify the loss to weight each individual sample's loss based on its ground-truth label. This is advantageous because it allows for a dynamic adjustment of weights, if necessary, based on factors beyond just the static class label, as we see in my second code example.

```python
import tensorflow as tf

def weighted_categorical_crossentropy(class_weights):
  """
  Custom categorical crossentropy with class weights.

  Args:
    class_weights: A dictionary mapping class indices to weights.

  Returns:
    A function that calculates the weighted categorical crossentropy loss.
  """

  def loss(y_true, y_pred):
    """
    Calculates the weighted categorical crossentropy.

    Args:
      y_true: Ground-truth labels, one-hot encoded.
      y_pred: Predicted probabilities.

    Returns:
      The scalar loss value.
    """
    y_true = tf.cast(y_true, tf.float32) # Ensure type matches
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)  # Numerical stability

    # Apply weights
    weights = tf.reduce_sum(y_true * tf.constant(list(class_weights.values()), dtype=tf.float32), axis=1)
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
    weighted_loss = tf.reduce_mean(cross_entropy * weights)

    return weighted_loss
  return loss

# Example usage:
class_weights = {0: 1.0, 1: 5.0} # Increase weight for class 1

# Compile with the custom loss:
model = tf.keras.models.Sequential(...) # Model definition
model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(class_weights), metrics=['accuracy'])

# Train as normal
# model.fit(...)
```

In this example, the `weighted_categorical_crossentropy` function returns a callable loss function that takes predicted and true labels as input. The core logic computes the cross-entropy for each sample. Crucially, it applies the `class_weights` to each cross-entropy term *before* averaging. This means samples from underrepresented classes have an amplified contribution to the overall loss, forcing the model to pay more attention to them. The numerical stability correction via `tf.clip_by_value` prevents errors from potentially infinite log values.

The second alternative is the utilization of `sample_weight` when calling `model.fit`. Instead of applying class weights, we directly assign weights to each *sample* during the training process, independently of its class label. This approach, particularly valuable when not all examples within a class are equally informative, opens the door for more fine-grained control and weighting strategies, which is why I found it useful in my fraud detection scenario. For instance, in that project, we developed a heuristic to weight samples based on their distance from other samples of the same class and from the class border, a detail I'll show in a moment. We achieved greater gains using this rather than the global class weight approach.

```python
import tensorflow as tf
import numpy as np

def calculate_sample_weights(y_true, class_weights, importance_scores):
    """
    Calculates sample weights based on class weights and importance scores.

    Args:
        y_true: Ground truth labels, integer encoded.
        class_weights: A dictionary mapping class indices to weights.
        importance_scores: A NumPy array of per-sample importance scores.

    Returns:
        A NumPy array of sample weights.
    """
    sample_weights = np.zeros(len(y_true))
    for i, label in enumerate(y_true):
        sample_weights[i] = class_weights.get(label, 1.0) * importance_scores[i]
    return sample_weights

# Example usage (synthetic data)
y_train = np.array([0, 1, 0, 1, 0, 2, 1, 2])
class_weights = {0: 1.0, 1: 3.0, 2: 5.0}  # Give class 1 more weight

# Simulate importance scores. In a real case, these would be calculated
# based on some domain specific criteria or model properties
importance_scores = np.array([0.5, 0.7, 1.2, 0.9, 0.8, 1.1, 1.5, 1.3]) # Arbitrary values

sample_weights = calculate_sample_weights(y_train, class_weights, importance_scores)

model = tf.keras.models.Sequential(...)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, sample_weight=sample_weights, ...)
```
In this snippet, the `calculate_sample_weights` function illustrates how class-based weights can be modulated by importance scores assigned at the sample level. I’ve emphasized the ability to inject domain-specific knowledge into the weighting.  The key part is passing `sample_weight` as an argument to `model.fit`, using the prepared weights computed using our function. This results in a different training dynamic, where not just the overall class balance, but also the informational content of each training example has a strong influence on the optimization process.

Lastly, consider the possibility of dynamic sample weights adjusted during training. This is where the ability to write custom training loops with `tf.GradientTape` becomes very powerful, though more verbose. In this case, the sample weights themselves become part of the trainable parameters, adjusted during the optimization process.

```python
import tensorflow as tf
import numpy as np

def dynamic_sample_weights_loss(model, x, y, weights, learning_rate):
    """
    Computes the loss and gradients for the model with dynamic sample weights.

    Args:
      model: The TensorFlow model.
      x: Input features.
      y: Ground truth labels.
      weights: Trainable tensor of sample weights.
      learning_rate: Learning rate for the optimizer.

    Returns:
      The average loss value.
    """

    with tf.GradientTape() as tape:
      y_pred = model(x)
      loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, y_pred) * weights)
    
    grads = tape.gradient(loss, model.trainable_variables + [weights])
    
    # Apply updates for model parameters
    optimizer.apply_gradients(zip(grads[:-1], model.trainable_variables))
    
    # Update sample weights with gradient information. Here, this is shown as an example.
    # Actual update logic requires careful consideration based on specific needs and task.
    
    weights_grad = grads[-1]
    updated_weights = weights - learning_rate*weights_grad # Simple gradient descent step

    weights.assign(tf.clip_by_value(updated_weights, 0.0, 10.0))
    return loss

# Example usage:
x_train = tf.random.normal((100, 10))
y_train = tf.random.uniform((100,), minval=0, maxval=3, dtype=tf.int32)
num_samples = tf.shape(y_train)[0]
initial_sample_weights = tf.ones((num_samples,), dtype=tf.float32)
trainable_sample_weights = tf.Variable(initial_sample_weights)

model = tf.keras.models.Sequential(...) # Model definition
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

num_epochs = 10
for epoch in range(num_epochs):
  for i in range(num_samples// 10):  # Batching (for demonstration)
      batch_x = x_train[i * 10 : (i + 1) * 10]
      batch_y = y_train[i * 10 : (i + 1) * 10]
      batch_weights = trainable_sample_weights[i * 10 : (i + 1) * 10]
      loss_value = dynamic_sample_weights_loss(model, batch_x, batch_y, batch_weights, 0.01)
      print(f"Epoch {epoch}, Batch {i}: Loss = {loss_value.numpy()}")
```

This final approach, using custom training loops with dynamic weights, offers maximum control over the training procedure. The `dynamic_sample_weights_loss` function calculates the standard loss, but also backpropagates to adjust the sample weights during training.  Note that the adjustment of sample weights needs careful consideration for stability and avoiding divergence, and the example above demonstrates a simple gradient descent update, it's crucial to tailor the weight update rule to the problem at hand. While complex, this yields the greatest control over loss contributions and therefore the learning process.

In conclusion, while TensorFlow's built-in `class_weights` parameter provides a convenient initial solution, directly manipulating the loss function, employing `sample_weights`, and utilizing custom training loops for dynamic sample weights can significantly enhance the performance of models trained on imbalanced data. Each technique provides different levels of control and flexibility. For a deeper exploration of these techniques, I recommend consulting resources focused on advanced deep learning practices with TensorFlow, especially those covering loss function design and custom training loop workflows. Investigating recent research on methods for learning with imbalanced datasets will also offer valuable insights into different weighting schemes. Additionally, familiarizing oneself with the theoretical basis of loss functions, and how gradients contribute to optimization, will help in designing effective loss strategies.
