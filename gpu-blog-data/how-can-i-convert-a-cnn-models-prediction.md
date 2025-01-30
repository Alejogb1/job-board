---
title: "How can I convert a CNN model's prediction to a probability?"
date: "2025-01-30"
id: "how-can-i-convert-a-cnn-models-prediction"
---
A convolutional neural network (CNN) often outputs raw scores, or logits, rather than probabilities. Directly interpreting these scores as probabilities is incorrect; they require a transformation. I've encountered this regularly while deploying image classification models for remote sensing projects and have found the softmax function to be the most common and effective method for converting CNN output to a probability distribution.

The core problem arises from the nature of CNN architectures. Typically, the final layer of a CNN consists of fully connected units, each corresponding to a distinct class. These units produce a numerical value that reflects how strongly the input image matches that class. However, these values can range anywhere, be negative, and are not inherently bounded between 0 and 1, nor do they sum to 1; the fundamental requirements for a proper probability distribution.

The softmax function addresses this. It takes a vector of raw scores (logits) and transforms it into a vector of probabilities where each probability represents the likelihood that the input image belongs to a particular class. The softmax function is defined mathematically as:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j))  for all j, where 'z' represents the logit vector.
```

Here, *z* is the input vector of logits. The function exponentiates each element of *z* using the natural base *e* and then divides each by the sum of all exponentiated values. This operation has several critical properties:

1.  **Non-negativity:** Since the exponential of any real number is positive, all output probabilities are greater than zero.
2.  **Normalization:** The sum of all elements in the output vector always equals 1 due to the normalization performed by the denominator.
3.  **Relative Ranking Preservation:** The softmax function preserves the relative ranking of the logits; the higher the logit, the higher the corresponding probability. However, the probability distribution emphasizes the largest scores by expanding differences.

This means the result becomes interpretable as a probability distribution across all classes. The class corresponding to the highest probability is considered the model's prediction.

I'll demonstrate this with Python code using the NumPy library, a standard tool in data science for numerical computations.

**Code Example 1: Basic Softmax Implementation**

```python
import numpy as np

def softmax(logits):
    """
    Converts a vector of logits to probabilities using the softmax function.

    Args:
      logits (np.ndarray): A 1D array of raw scores (logits).

    Returns:
      np.ndarray: A 1D array of probabilities.
    """
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities

# Example Usage
raw_scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(raw_scores)
print(f"Logits: {raw_scores}")
print(f"Probabilities: {probabilities}")
print(f"Sum of probabilities: {np.sum(probabilities)}")

# Another Example
raw_scores_2 = np.array([-2, 1.5, 0.5, -1])
probabilities_2 = softmax(raw_scores_2)
print(f"Logits 2: {raw_scores_2}")
print(f"Probabilities 2: {probabilities_2}")
print(f"Sum of probabilities 2: {np.sum(probabilities_2)}")
```

In this first example, `softmax` takes a NumPy array of logits and returns corresponding probabilities. The `np.exp` function handles the exponentiation and the division normalizes the resulting values so the sum equals one. We print both the initial raw scores and their resulting probabilities for demonstration. Note that while the rank order is preserved in both the logit vector and probability vector, the differences are expanded by softmax.

**Code Example 2: Softmax with Batch Handling**

Often, you'll process multiple images simultaneously (a batch) to improve performance. This requires handling multi-dimensional input.

```python
import numpy as np

def batch_softmax(logits):
    """
    Converts a batch of logits to probabilities using the softmax function.

    Args:
      logits (np.ndarray): A 2D array where each row is a vector of logits.
      (batch_size, num_classes).

    Returns:
      np.ndarray: A 2D array of probabilities, same shape as logits.
    """
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities


# Example Usage: Batch of 3 Images with 4 Classes each
batch_logits = np.array([
    [2.0, 1.0, 0.1, -0.5],
    [-1, 0.5, 2.2, 0],
    [0.1, 0.4, -1, 1]
])
batch_probabilities = batch_softmax(batch_logits)
print(f"Batch Logits:\n{batch_logits}")
print(f"Batch Probabilities:\n{batch_probabilities}")

# Verify that each row (probability vector) sums to approximately 1
for prob in batch_probabilities:
  print(f"Sum of individual prob vector: {np.sum(prob)}")
```

This second example, `batch_softmax`, modifies the first example to work with a 2D array where each row represents the logits for an individual image in a batch. We use `axis=1` in the `np.sum` call to sum across the class scores for each image and `keepdims=True` to preserve dimensionality for proper broadcasting. This ensures each vector is normalized independently. The sum of probabilities per image should approach 1.

**Code Example 3: Handling Numerical Instability**

In some scenarios, large positive logits can result in overflow during exponentiation, yielding `inf`. Subtracting the maximum value of the logits before exponentiation stabilizes the operation without changing the result (as it's in the exponential).

```python
import numpy as np

def stable_softmax(logits):
    """
    Converts logits to probabilities with stable numerical operations.

    Args:
      logits (np.ndarray): A 1D or 2D array of raw scores (logits).

    Returns:
      np.ndarray: Probabilities, same shape as logits.
    """
    logits = np.array(logits)  # Ensure it's a numpy array
    if logits.ndim == 1: # Handle single logit
      max_logit = np.max(logits)
      shifted_logits = logits - max_logit
      exp_shifted_logits = np.exp(shifted_logits)
      probabilities = exp_shifted_logits / np.sum(exp_shifted_logits)
    elif logits.ndim == 2: # Handle batched logit
      max_logits = np.max(logits, axis=1, keepdims=True)
      shifted_logits = logits - max_logits
      exp_shifted_logits = np.exp(shifted_logits)
      probabilities = exp_shifted_logits / np.sum(exp_shifted_logits, axis=1, keepdims=True)
    else:
      raise ValueError("logits must be 1D or 2D")
    return probabilities

# Example usage:
unstable_logits = np.array([1000, 999, 800])
probabilities_unstable = softmax(unstable_logits)
print(f"Unstable softmax, potentially erroneous: {probabilities_unstable}")

stable_probabilities = stable_softmax(unstable_logits)
print(f"Stable softmax, correct: {stable_probabilities}")

# Example with batch of logits
batch_logits_unstable = np.array([[1000, 999, 800], [500, 400, 450]])
probabilities_unstable = batch_softmax(batch_logits_unstable)
print(f"Unstable batch softmax, potentially erroneous:\n{probabilities_unstable}")

probabilities_stable = stable_softmax(batch_logits_unstable)
print(f"Stable batch softmax, correct:\n{probabilities_stable}")
```

Here, `stable_softmax` subtracts the maximum value of the logits before the exponentiation, preventing numerical instability. This is particularly useful when working with models where output layers might result in large values. The example demonstrates the difference in the raw softmax function versus the stable version, especially with large logit values.

In conclusion, while your CNN model's output gives an indication of class assignment confidence via raw scores, converting these scores into probabilities using a stable implementation of the softmax function is crucial for a proper interpretation of the model's predictions. The softmax output provides a normalized, positive distribution suitable for use in subsequent tasks such as calculating classification metrics, performing uncertainty estimation, and making decisions based on probability thresholds.

For further understanding, I recommend consulting resources on:

*   Numerical computation techniques for machine learning, which cover common strategies for handling issues like numerical overflow.
*   The fundamentals of probability distributions and how they apply to machine learning model output.
*   Detailed documentation on your chosen machine learning framework (such as TensorFlow or PyTorch) which often provides optimized and numerically stable implementations of the softmax function.
