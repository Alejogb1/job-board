---
title: "Why is computing loss from logits numerically stable?"
date: "2025-01-30"
id: "why-is-computing-loss-from-logits-numerically-stable"
---
The numerical stability of computing loss from logits stems directly from the inherent properties of the softmax function and the cross-entropy loss function, particularly their interaction in preventing exponent overflow and underflow.  In my experience optimizing deep learning models, I've observed firsthand how neglecting this crucial detail can lead to training instability and inaccurate gradients, hindering model convergence.

**1. A Clear Explanation:**

Logits represent the unnormalized pre-activation values from the final layer of a classification neural network.  Directly applying these logits to calculate probabilities using the naive exponential approach (e<sup>logit</sup> / Σe<sup>logit</sup>) is computationally risky.  Extremely large logits lead to exponent overflow, resulting in `inf` values, while extremely small logits cause exponent underflow, producing `0` values. Both scenarios cripple the subsequent loss calculation and gradient computation, leading to `NaN` (Not a Number) errors or completely stalled training.

The solution lies in the careful combination of the softmax function and the cross-entropy loss.  The softmax function, defined as:

softmax(z<sub>i</sub>) = e<sup>z<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>

where z<sub>i</sub> are the logits, normalizes these values into probabilities. However, the direct computation still suffers from the aforementioned numerical issues. The key to stability is found in how we calculate the cross-entropy loss.  Cross-entropy loss for a single sample is typically defined as:

L = - Σ<sub>i</sub> y<sub>i</sub> log(softmax(z<sub>i</sub>))

where y<sub>i</sub> is 1 if class i is the correct class and 0 otherwise (one-hot encoding).  Direct computation remains vulnerable.  Instead, we employ a numerically stable formulation by leveraging the logarithmic properties.  We rewrite the equation as:

L = - Σ<sub>i</sub> y<sub>i</sub> (z<sub>i</sub> - log(Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>))

This seemingly simple algebraic manipulation is profoundly important.  We subtract the log-sum-exp (logsumexp) term, log(Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>), from each logit.  This operation, which is commonly computed using the numerically stable logsumexp function, prevents overflow and underflow.  The logsumexp function itself utilizes a clever trick:

logsumexp(z) = max(z) + log(Σ<sub>j</sub> e<sup>z<sub>j</sub> - max(z)</sup>)

By subtracting the maximum logit value from all logits before exponentiation, we avoid generating excessively large or small exponentials. This ensures the summation remains within a manageable numerical range, mitigating the overflow/underflow problem. The final addition of max(z) compensates for the initial subtraction.

**2. Code Examples with Commentary:**

Here are three Python examples demonstrating different approaches, highlighting the importance of numerical stability:

**Example 1:  Unstable Implementation:**

```python
import numpy as np

def unstable_cross_entropy(logits, labels):
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    loss = -np.sum(labels * np.log(probabilities))
    return loss

logits = np.array([1000, 10, 1])  # Example with large logits leading to overflow
labels = np.array([1, 0, 0])
loss = unstable_cross_entropy(logits, labels)
print(f"Unstable Loss: {loss}") #Output: inf or NaN

```

This example directly implements the naive softmax and cross-entropy calculation.  With large logits, it will almost certainly result in `inf` or `NaN`.

**Example 2: Stable Implementation using Logsumexp:**

```python
import numpy as np

def logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

def stable_cross_entropy(logits, labels):
    lse = logsumexp(logits)
    loss = np.sum(labels * (lse - logits))
    return loss

logits = np.array([1000, 10, 1])
labels = np.array([1, 0, 0])
loss = stable_cross_entropy(logits, labels)
print(f"Stable Loss: {loss}") # Output: A finite, numerically stable value.
```

This example uses the logsumexp trick to compute the cross-entropy loss stably. It correctly handles the large logits, producing a finite result.

**Example 3:  TensorFlow/Keras Implementation:**

```python
import tensorflow as tf

logits = tf.constant([1000., 10., 1.])
labels = tf.constant([1., 0., 0.])

loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
print(f"TensorFlow Loss: {loss.numpy()}") # Output: A finite, numerically stable value
```

TensorFlow/Keras's built-in `categorical_crossentropy` function with `from_logits=True` automatically handles the numerical stability issues, leveraging optimized implementations similar to the logsumexp approach.  Using established frameworks is highly recommended for production environments, as they incorporate various optimizations and safeguards.

**3. Resource Recommendations:**

I'd suggest consulting standard deep learning textbooks, focusing on sections on numerical stability and loss function derivations.  Furthermore, reviewing the source code of popular deep learning frameworks like TensorFlow or PyTorch can provide valuable insight into how these frameworks address this critical issue.  Finally, exploring research papers on numerical optimization techniques within the context of deep learning would provide a more comprehensive understanding.  Examining papers focusing on the specifics of the softmax and cross-entropy loss functions will be particularly useful.  These materials offer detailed mathematical explanations and practical implementations for stable computations.
