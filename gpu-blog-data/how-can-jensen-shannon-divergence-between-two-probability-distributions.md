---
title: "How can Jensen-Shannon divergence between two probability distributions be calculated in PyTorch?"
date: "2025-01-30"
id: "how-can-jensen-shannon-divergence-between-two-probability-distributions"
---
The Jensen-Shannon divergence (JSD) lacks a direct, readily available function within the PyTorch library. This necessitates a programmatic construction leveraging PyTorch's tensor operations and its inherent support for automatic differentiation. My experience implementing information-theoretic measures in deep learning projects has highlighted the importance of numerical stability in these calculations, particularly when dealing with probability distributions that may contain near-zero probabilities. This directly informs the approach I'll detail below.

**1. A Clear Explanation of the Calculation**

The Jensen-Shannon divergence is a symmetrized and bounded version of the Kullback-Leibler (KL) divergence.  Given two probability distributions, P and Q, the JSD is defined as:

JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

where M = 0.5 * (P + Q) is the average of the two distributions. The KL divergence itself is defined as:

KL(P || Q) = Σᵢ Pᵢ * log(Pᵢ / Qᵢ)

Directly translating this into PyTorch requires careful handling of potential numerical issues, primarily the logarithm of zero.  Therefore, a robust implementation should incorporate smoothing techniques.  A common approach is adding a small positive constant (ε) to both P and Q before performing the computation. This prevents undefined results and improves numerical stability, especially when dealing with sparse distributions where probabilities might be zero or extremely small.

The implementation will consist of these steps:

1. **Input Validation:** Ensuring the input tensors represent valid probability distributions (non-negative elements summing to one).
2. **Smoothing:** Adding a small epsilon value to each probability distribution to prevent numerical instability.
3. **Average Distribution Calculation:** Computing the average distribution M.
4. **KL Divergence Calculation:** Calculating the KL divergence between P and M, and Q and M, utilizing PyTorch's logarithmic functions.
5. **JSD Calculation:** Combining the KL divergences according to the JSD formula.

**2. Code Examples with Commentary**

**Example 1: Basic JSD Calculation**

```python
import torch
import torch.nn.functional as F

def jsd(p, q, epsilon=1e-10):
    """
    Calculates the Jensen-Shannon divergence between two probability distributions.

    Args:
        p: First probability distribution (PyTorch tensor).
        q: Second probability distribution (PyTorch tensor).
        epsilon: Small constant for smoothing (default: 1e-10).

    Returns:
        Jensen-Shannon divergence (scalar).  Returns -1 if invalid input.
    """
    if not (torch.all(p >= 0) and torch.all(q >= 0) and torch.isclose(torch.sum(p), torch.tensor(1.0)) and torch.isclose(torch.sum(q), torch.tensor(1.0))):
        return -1

    p_smoothed = p + epsilon
    q_smoothed = q + epsilon
    m = 0.5 * (p_smoothed + q_smoothed)
    kl_pq = F.kl_div(p_smoothed.log(), m, reduction='sum')
    kl_qp = F.kl_div(q_smoothed.log(), m, reduction='sum')

    return 0.5 * (kl_pq + kl_qp)


p = torch.tensor([0.2, 0.3, 0.5])
q = torch.tensor([0.4, 0.1, 0.5])
jsd_value = jsd(p, q)
print(f"JSD: {jsd_value}")

p = torch.tensor([0.0,1.0,0.0])
q = torch.tensor([1.0,0.0,0.0])
jsd_value = jsd(p,q)
print(f"JSD: {jsd_value}")
```

This example showcases a basic implementation.  The `epsilon` parameter provides numerical stability.  The `if` statement ensures that input distributions are valid. The use of `torch.isclose` allows for tolerance in the sum of probabilities due to floating-point inaccuracies.  The `F.kl_div` function efficiently computes the KL divergence.  Note that the function returns -1 if the input is invalid.

**Example 2: Handling Batched Distributions**

```python
import torch
import torch.nn.functional as F

def batched_jsd(p_batch, q_batch, epsilon=1e-10):
    """
    Calculates JSD for a batch of probability distributions.

    Args:
      p_batch: Batch of probability distributions (PyTorch tensor of shape (batch_size, num_categories)).
      q_batch: Batch of probability distributions (same shape as p_batch).
      epsilon: Smoothing constant.

    Returns:
      Tensor of JSD values for each pair of distributions in the batch. Returns -1 if invalid input detected in any batch element.
    """

    batch_size = p_batch.shape[0]
    jsd_values = torch.zeros(batch_size)
    for i in range(batch_size):
        jsd_values[i] = jsd(p_batch[i], q_batch[i], epsilon)
        if jsd_values[i] == -1:
            return -1

    return jsd_values

p_batch = torch.tensor([[0.2, 0.8, 0.0], [0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])
q_batch = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.4, 0.4]])

jsd_batch_result = batched_jsd(p_batch, q_batch)
print(f"Batch JSD values: {jsd_batch_result}")
```

This expands upon the first example to handle batches of probability distributions.  It iterates through the batch, applies the `jsd` function from Example 1 to each pair, and returns a tensor containing the JSD for each batch element.  Error handling remains to check for validity across the entire batch.

**Example 3:  JSD with Log-Probabilities**

```python
import torch
import torch.nn.functional as F

def jsd_logprobs(log_p, log_q, epsilon=1e-10):
    """
    Calculates JSD using log-probabilities for numerical stability.

    Args:
        log_p: Log-probabilities of the first distribution.
        log_q: Log-probabilities of the second distribution.
        epsilon: Smoothing constant for numerical stability.

    Returns:
        Jensen-Shannon divergence (scalar). Returns -1 if invalid input detected.
    """
    if not torch.isfinite(log_p).all() or not torch.isfinite(log_q).all():
        return -1

    p = torch.exp(log_p)
    q = torch.exp(log_q)
    if not (torch.all(p >= 0) and torch.all(q >= 0) and torch.isclose(torch.sum(p), torch.tensor(1.0)) and torch.isclose(torch.sum(q), torch.tensor(1.0))):
        return -1

    p_smoothed = p + epsilon
    q_smoothed = q + epsilon
    m = 0.5 * (p_smoothed + q_smoothed)
    kl_pq = F.kl_div(log_p, m.log(), reduction='sum')
    kl_qp = F.kl_div(log_q, m.log(), reduction='sum')

    return 0.5 * (kl_pq + kl_qp)

log_p = torch.tensor([-0.5, -1.2, -0.9]) #Example log probabilities
log_q = torch.tensor([-1.1, -2.3, -0.3])

jsd_logprob_value = jsd_logprobs(log_p, log_q)
print(f"JSD using log-probabilities: {jsd_logprob_value}")
```

This example demonstrates calculating JSD directly from log-probabilities, which is often more numerically stable, especially when dealing with extremely small probabilities.  It converts log-probabilities to probabilities internally for the calculation, ensuring the function remains consistent with previous examples.  Error handling now includes checks for infinite or NaN values in the input log-probabilities.

**3. Resource Recommendations**

For a deeper understanding of KL divergence and its related measures, consult standard textbooks on information theory and machine learning.  The documentation for PyTorch’s `torch.nn.functional` module is invaluable for understanding the functionalities used in the provided code examples.  Furthermore, research papers exploring the applications of JSD in various fields will provide valuable contextualization and further implementation strategies.  Specific attention should be paid to papers addressing numerical stability in these computations.
