---
title: "How can a custom loss function be used for out-of-distribution detection with CNNs in TensorFlow 2.0+?"
date: "2025-01-30"
id: "how-can-a-custom-loss-function-be-used"
---
Out-of-distribution (OOD) detection is a crucial yet often overlooked aspect of deploying Convolutional Neural Networks (CNNs).  Standard classification accuracy metrics, while informative for in-distribution performance, fail to adequately characterize a model's robustness against unseen data.  My experience developing anomaly detection systems for industrial image processing highlighted this limitation – high accuracy on training data did not translate to reliable performance on anomalous inputs.  Therefore, crafting a custom loss function tailored for OOD detection offers a powerful alternative to post-hoc techniques. This approach leverages the inherent learning process of the CNN, encouraging the model to not only classify in-distribution samples but also to explicitly learn to differentiate them from outliers.

The core principle underlying this approach lies in formulating a loss function that penalizes the network for assigning high confidence scores to OOD samples.  Instead of solely focusing on minimizing cross-entropy loss between predicted and true labels for in-distribution data, we augment the loss function to explicitly incorporate a term that encourages lower confidence scores for OOD examples.  This necessitates a clearly defined OOD dataset during training.  This allows for simultaneous learning of both in-distribution classification and OOD discrimination within the same training loop, increasing efficiency compared to many post-hoc methods.  Several strategies can achieve this augmentation, each with its strengths and weaknesses.  I'll present three examples below.


**1.  Confidence Penalty Loss:**

This method introduces a penalty proportional to the maximum predicted probability for OOD samples.  The rationale is simple:  a well-calibrated model should assign low confidence scores to inputs it's not trained to recognize.

```python
import tensorflow as tf

def confidence_penalty_loss(y_true, y_pred, ood_mask, lambda_ood=0.1):
  """
  Custom loss function with a confidence penalty for OOD samples.

  Args:
    y_true: True labels (one-hot encoded).
    y_pred: Predicted probabilities.
    ood_mask: Binary mask indicating OOD samples (1 for OOD, 0 for ID).
    lambda_ood: Weighting factor for the OOD penalty.

  Returns:
    Total loss (cross-entropy + OOD penalty).
  """

  cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
  ood_penalty = tf.reduce_mean(tf.boolean_mask(tf.reduce_max(y_pred, axis=1), ood_mask))
  total_loss = cross_entropy_loss + lambda_ood * ood_penalty
  return total_loss

# Example usage:
model = tf.keras.models.Sequential(...) # Your CNN model
model.compile(optimizer='adam', loss=lambda y_true, y_pred: confidence_penalty_loss(y_true, y_pred, ood_mask)) # ood_mask needs to be provided during training
model.fit(...)
```

This code defines a `confidence_penalty_loss` function which adds a penalty term to the standard categorical cross-entropy loss.  The `ood_mask` is crucial; it acts as a binary selector, applying the penalty only to OOD instances. The `lambda_ood` hyperparameter controls the strength of the OOD penalty. Tuning this parameter is vital; a high value may lead to over-penalization and affect in-distribution performance.  In my experience with industrial imagery, a carefully tuned `lambda_ood` was essential for optimal OOD detection without sacrificing in-distribution accuracy.


**2.  Margin-based Loss:**

This approach aims to create a larger margin between the confidence scores of in-distribution and OOD samples.  We modify the loss to specifically penalize OOD samples with confidence scores exceeding a predefined threshold.

```python
import tensorflow as tf

def margin_based_loss(y_true, y_pred, ood_mask, margin=0.5):
  """
  Custom loss function enforcing a margin between ID and OOD confidence scores.

  Args:
    y_true: True labels (one-hot encoded).
    y_pred: Predicted probabilities.
    ood_mask: Binary mask indicating OOD samples (1 for OOD, 0 for ID).
    margin: Desired margin between ID and OOD confidence scores.

  Returns:
    Total loss (cross-entropy + margin penalty).
  """

  cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
  ood_confidence = tf.boolean_mask(tf.reduce_max(y_pred, axis=1), ood_mask)
  margin_penalty = tf.reduce_mean(tf.maximum(0.0, ood_confidence - margin))
  total_loss = cross_entropy_loss + margin_penalty
  return total_loss

# Example Usage
model = tf.keras.models.Sequential(...) # Your CNN model
model.compile(optimizer='adam', loss=lambda y_true, y_pred: margin_based_loss(y_true, y_pred, ood_mask)) # ood_mask needs to be provided during training
model.fit(...)
```

Here, `margin_based_loss` adds a penalty if the maximum predicted probability for an OOD sample exceeds the `margin`. The `tf.maximum(0.0, ood_confidence - margin)` ensures that the penalty is only applied when the confidence surpasses the margin.  This approach directly encourages a clearer separation in predicted confidence between in-distribution and out-of-distribution data. Setting the `margin` requires careful consideration and potentially experimentation.

**3.  Energy-based Loss:**

This loss function leverages the concept of energy-based models.  We assume that the model's output represents the energy of the input; lower energy implies higher probability of being in-distribution.  The loss function aims to minimize the energy for in-distribution samples and maximize it for OOD samples.


```python
import tensorflow as tf

def energy_based_loss(y_true, y_pred, ood_mask, beta=1.0):
  """
  Custom loss function based on energy-based model principles.

  Args:
    y_true: True labels (one-hot encoded).
    y_pred: Predicted probabilities.  Should be interpreted as energy.
    ood_mask: Binary mask indicating OOD samples (1 for OOD, 0 for ID).
    beta: Weighting factor for OOD samples.

  Returns:
    Total loss.
  """

  in_distribution_energy = tf.reduce_mean(tf.boolean_mask(-tf.reduce_max(y_pred, axis=1), tf.logical_not(ood_mask)))
  ood_energy = tf.reduce_mean(tf.boolean_mask(tf.reduce_max(y_pred, axis=1), ood_mask))

  total_loss = -in_distribution_energy + beta * ood_energy
  return total_loss


# Example Usage
model = tf.keras.models.Sequential(...) # Your CNN model.  Output layer should be a single neuron.
model.compile(optimizer='adam', loss=lambda y_true, y_pred: energy_based_loss(y_true, y_pred, ood_mask)) # ood_mask needs to be provided during training
model.fit(...)
```

Note that in this implementation, the output layer of the CNN should be modified to produce a single scalar value (energy). This approach differs significantly from the previous two, requiring a conceptual shift in how the model's output is interpreted.  The `beta` parameter controls the relative importance of the OOD energy term.


**Resource Recommendations:**

For a deeper understanding, I recommend exploring the relevant chapters in "Deep Learning" by Goodfellow et al. and  research papers on energy-based models and anomaly detection. Examining TensorFlow's official documentation on custom training loops and loss functions is also essential.  Furthermore, reviewing papers specifically addressing OOD detection in the context of CNNs will prove invaluable.  Understanding the limitations of each approach is crucial for selecting and tuning an appropriate custom loss function for your specific OOD detection task.  Careful consideration of the data distribution, model architecture and hyperparameter tuning are key to successful implementation.
