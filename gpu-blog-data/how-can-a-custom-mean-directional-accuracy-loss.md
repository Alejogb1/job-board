---
title: "How can a custom mean directional accuracy loss function be implemented in Keras?"
date: "2025-01-30"
id: "how-can-a-custom-mean-directional-accuracy-loss"
---
The core challenge in implementing a custom mean directional accuracy loss function in Keras lies in effectively representing directional accuracy as a differentiable quantity suitable for backpropagation.  My experience developing similar loss functions for object detection and pose estimation highlighted the need for careful consideration of both the metric's definition and its numerical stability within the gradient descent framework.  Directly applying a discrete accuracy measure proves problematic; instead, a continuous approximation reflecting the angular deviation between predicted and ground truth directions is required.

**1.  Explanation:**

A conventional mean accuracy metric is unsuitable for directional data because it ignores the magnitude of the error.  Two predictions, one slightly and one significantly off the true direction, would both register as incorrect.  Mean directional accuracy, on the other hand, assesses the closeness of predicted vectors to ground truth vectors, focusing on angular discrepancies.  We aim to minimize the average angular error across all samples.

The process involves several steps:

a) **Vector Representation:** Both predicted and ground truth directional data must be represented as vectors.  This could be Cartesian coordinates (x, y, z), spherical coordinates (r, θ, φ), or any other suitable representation dependent on the specific application.

b) **Angular Difference Calculation:**  The angular difference between the predicted and ground truth vectors is computed using the dot product.  Given two vectors *v*<sub>predicted</sub> and *v*<sub>ground truth</sub>, the cosine of the angle θ between them is:

cos(θ) = (*v*<sub>predicted</sub> • *v*<sub>ground truth</sub>) / (||*v*<sub>predicted</sub>|| ||*v*<sub>ground truth</sub>||)

where ||*v*|| represents the magnitude of vector *v*.  The angle θ can then be obtained using the arccosine function:

θ = arccos(cos(θ))

This angle, representing the directional error, serves as the basis for the loss function.

c) **Loss Function Definition:**  To ensure differentiability,  the loss function should operate directly on the angle θ, rather than on a discrete classification of 'correct' or 'incorrect'.  A suitable choice is a differentiable function that penalizes larger angular deviations more strongly. I've found the following functions effective in practice:

* **Absolute Angular Error:**  This is a simple and effective choice.  The loss is simply the absolute value of the angular difference:  L = |θ|.  Its derivative is straightforward to compute.

* **Squared Angular Error:** This function emphasizes larger errors.  L = θ².  The derivative is also easily calculated.

* **Weighted Angular Error:** This allows for flexibility in handling various angular deviations. For instance, small angular errors could be less penalized compared to large ones.  One example is using a weighted function such as L = θ² if θ > π/4, else L = θ/2.

d) **Mean Calculation:** The mean angular error across all samples is then computed, producing the final loss value used by the optimizer.


**2. Code Examples with Commentary:**

**Example 1: Absolute Angular Error with Cartesian Coordinates**

```python
import tensorflow as tf
import keras.backend as K

def directional_accuracy_loss(y_true, y_pred):
    """
    Computes the mean absolute angular error between predicted and ground truth vectors.

    Args:
        y_true: Tensor of ground truth vectors (shape: (batch_size, 3)).
        y_pred: Tensor of predicted vectors (shape: (batch_size, 3)).

    Returns:
        The mean absolute angular error.
    """
    dot_product = K.sum(y_true * y_pred, axis=-1)
    magnitudes_true = K.sqrt(K.sum(K.square(y_true), axis=-1))
    magnitudes_pred = K.sqrt(K.sum(K.square(y_pred), axis=-1))
    cos_theta = dot_product / (magnitudes_true * magnitudes_pred + K.epsilon()) #Adding epsilon for numerical stability
    theta = K.acos(K.clip(cos_theta, -1.0, 1.0)) #Clipping to handle numerical inaccuracies.
    return K.mean(theta)

model.compile(loss=directional_accuracy_loss, optimizer='adam')
```

This example uses Cartesian coordinates and the absolute angular error. Note the addition of `K.epsilon()` to prevent division by zero and `K.clip` to handle potential numerical issues with `arccos`.

**Example 2: Squared Angular Error with Spherical Coordinates**

```python
import tensorflow as tf
import keras.backend as K

def spherical_directional_loss(y_true, y_pred):
    """
    Computes the mean squared angular error between predicted and ground truth vectors using spherical coordinates.  Assumes y_true and y_pred are in (theta, phi) format.

    Args:
        y_true: Tensor of ground truth vectors (shape: (batch_size, 2)).
        y_pred: Tensor of predicted vectors (shape: (batch_size, 2)).

    Returns:
        The mean squared angular error.
    """
    theta_diff = y_true[:,0] - y_pred[:,0]
    phi_diff = y_true[:,1] - y_pred[:,1]
    squared_error = K.square(theta_diff) + K.square(phi_diff) #Simplified squared angular error, assuming small angular differences.  A more rigorous approach may be needed for large differences
    return K.mean(squared_error)


model.compile(loss=spherical_directional_loss, optimizer='adam')
```

This example demonstrates the use of spherical coordinates and squared angular error, simplifying the angular difference calculation by assuming small deviations. A more comprehensive calculation considering the curvature of the sphere might be needed for larger angular errors.

**Example 3: Weighted Angular Error**


```python
import tensorflow as tf
import keras.backend as K

def weighted_directional_loss(y_true, y_pred):
  """
  Computes a weighted mean angular error.

  Args:
      y_true: Tensor of ground truth vectors (shape: (batch_size, 3)).
      y_pred: Tensor of predicted vectors (shape: (batch_size, 3)).

  Returns:
      The weighted mean angular error.
  """
  dot_product = K.sum(y_true * y_pred, axis=-1)
  magnitudes_true = K.sqrt(K.sum(K.square(y_true), axis=-1))
  magnitudes_pred = K.sqrt(K.sum(K.square(y_pred), axis=-1))
  cos_theta = dot_product / (magnitudes_true * magnitudes_pred + K.epsilon())
  theta = K.acos(K.clip(cos_theta, -1.0, 1.0))
  weighted_error = K.switch(theta > K.constant(0.785), K.square(theta), theta /2) # Weighting based on a threshold (π/4 radians)
  return K.mean(weighted_error)

model.compile(loss=weighted_directional_loss, optimizer='adam')

```

This example introduces a weighted angular error, applying different penalty based on the magnitude of angular difference.  This exemplifies  the flexibility afforded by defining a custom loss.


**3. Resource Recommendations:**

For deeper understanding of numerical stability in deep learning, consult standard textbooks on numerical optimization and machine learning.  Furthermore,  reviewing publications on differentiable rendering and differentiable physics simulators will provide insights into handling similar challenges in representing continuous approximations of discrete quantities for gradient-based optimization.  Finally, explore the Keras documentation for comprehensive details on custom loss function implementation and best practices.  Careful consideration of these resources will help address potential issues, especially in dealing with edge cases and numerical instability.
