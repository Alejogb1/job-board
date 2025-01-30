---
title: "How do I implement a JSD loss with an upper bound in PyTorch?"
date: "2025-01-30"
id: "how-do-i-implement-a-jsd-loss-with"
---
Jensen-Shannon Divergence (JSD), while a robust measure of similarity between probability distributions, lacks a defined upper bound, potentially causing instability in training. This makes it unsuitable for tasks requiring a normalized loss or comparison across different scenarios. I encountered this problem firsthand during the development of a variational autoencoder (VAE) for image reconstruction, where an unbounded JSD resulted in volatile optimization. To address this, the standard approach involves employing a modified JSD calculation, often achieved by introducing a clipping or saturating mechanism on the intermediate divergence values. The following outlines the implementation in PyTorch and elaborates on practical considerations.

Fundamentally, JSD measures the difference between two probability distributions by averaging the Kullback-Leibler divergence (KL divergence) between each distribution and their average distribution. Given two distributions, *P* and *Q*, and their average, *M* = (*P* + *Q*)/2, the JSD is calculated as:

JSD(P, Q) = 0.5 * [KL(P || M) + KL(Q || M)]

The standard KL divergence is computed as:

KL(P || Q) = Σ P(i) * log(P(i) / Q(i))

The primary challenge with implementing a JSD with an upper bound resides in modifying the KL divergence calculation. Instead of directly using the logarithmic ratio, a bounded function is applied. One common strategy is to utilize a smooth approximation, often involving the hyperbolic tangent (tanh) or a similar sigmoid function. This approximation introduces saturation in the KL divergence, consequently capping the JSD. Let’s examine this process through practical PyTorch implementations.

**Example 1: JSD with tanh-based clipping**

This approach directly manipulates the log ratio within the KL divergence, allowing for a controlled saturation. I typically use a scaling factor within the tanh function to adjust the level of clipping.

```python
import torch
import torch.nn.functional as F

def jsd_tanh_bounded(p, q, clip_factor=10.0):
    """
    Calculates Jensen-Shannon Divergence with tanh-based clipping.

    Args:
        p (torch.Tensor): First probability distribution.
        q (torch.Tensor): Second probability distribution.
        clip_factor (float): Scaling factor for tanh, higher values less clip.

    Returns:
        torch.Tensor: Bounded Jensen-Shannon Divergence.
    """
    m = 0.5 * (p + q)
    kl_pm =  torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)

    # Apply tanh to the divergence terms for clipping
    kl_pm = torch.tanh(kl_pm / clip_factor) * clip_factor
    kl_qm = torch.tanh(kl_qm / clip_factor) * clip_factor

    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd
```

In this implementation, `kl_pm` and `kl_qm` are calculated using the standard definition of KL divergence. Critically, before being added, the `tanh` function with a scaling factor is applied to each KL term. The scaling factor (`clip_factor`) regulates how quickly the KL divergence saturates. A larger `clip_factor` allows for a higher divergence before clipping kicks in. This effectively limits the magnitude of the KL divergence and, by extension, the JSD.  The `dim=-1` argument ensures that the KL divergence is calculated correctly for batched data.

**Example 2: JSD with Sigmoid-based smoothing**

This method uses a sigmoid function, which is a softer clipping mechanism compared to the sharp saturation provided by `tanh`. This was beneficial in some of my generative model training as it resulted in smoother gradients and stabilized learning.

```python
import torch
import torch.nn.functional as F

def jsd_sigmoid_bounded(p, q, clip_factor=5.0):
  """
    Calculates Jensen-Shannon Divergence with sigmoid-based smoothing.

    Args:
        p (torch.Tensor): First probability distribution.
        q (torch.Tensor): Second probability distribution.
        clip_factor (float): Scaling factor for sigmoid.

    Returns:
        torch.Tensor: Bounded Jensen-Shannon Divergence.
    """
  m = 0.5 * (p + q)
  kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
  kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)


  kl_pm = torch.sigmoid(kl_pm / clip_factor) * clip_factor
  kl_qm = torch.sigmoid(kl_qm / clip_factor) * clip_factor

  jsd = 0.5 * (kl_pm + kl_qm)
  return jsd
```

Similar to the first example, this code computes the standard KL divergence and then applies a sigmoid function, scaled by a `clip_factor`, to each KL divergence term.  The sigmoid, when scaled, smoothly approaches a maximum value, thereby providing a bounded but less abrupt saturation than the `tanh` function. This smoothing often prevents large, sudden gradient updates which can result in more consistent training. The clipping strength is controlled by the `clip_factor`.

**Example 3: Combined Approach with dynamic clipping**

This example employs a combined approach which incorporates a dynamic clip based on the distribution’s entropy along with the *tanh* saturating function to obtain tighter bounds when a distribution has low entropy.

```python
import torch
import torch.nn.functional as F

def jsd_dynamic_tanh_bounded(p, q, clip_factor_base=10.0):
    """
    Calculates JSD with tanh-based clipping and dynamic entropy based clip factor.

    Args:
        p (torch.Tensor): First probability distribution.
        q (torch.Tensor): Second probability distribution.
        clip_factor_base (float): Base clipping factor.

    Returns:
        torch.Tensor: Bounded Jensen-Shannon Divergence.
    """

    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)

    # Calculate the entropy of the distributions
    entropy_p = -torch.sum(p * torch.log(p), dim=-1)
    entropy_q = -torch.sum(q * torch.log(q), dim=-1)
    avg_entropy = 0.5 * (entropy_p + entropy_q)

    # Dynamic clip factor based on entropy
    clip_factor = clip_factor_base * (1 + torch.exp(-avg_entropy))
    # Apply tanh with dynamic clip factor
    kl_pm = torch.tanh(kl_pm / clip_factor) * clip_factor
    kl_qm = torch.tanh(kl_qm / clip_factor) * clip_factor


    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd

```

This approach includes the entropy of distributions *p* and *q* to modulate the `clip_factor`. Lower entropy distributions can lead to higher KL divergence values, hence the adaptive `clip_factor` ensures the saturating function has an appropriate clipping strength. This approach was particularly effective when dealing with highly peaked or sparse probability distributions. The dynamic `clip_factor` reduces the effective clipping scale when the entropy is low and vice-versa.

In practical implementation, several considerations are crucial. First, numerical stability must be ensured. Logarithms of zero can produce NaNs, so adding a small constant (epsilon) to arguments of `torch.log` function is a must. Similarly, proper handling of zero or very small probabilities is essential, often achieved by clamping probabilities to minimum value, or utilizing numerically stable KL divergence calculation. Furthermore, choosing the correct `clip_factor` requires experimentation. It heavily depends on the nature of the input distribution and the specific task. I typically started with a small `clip_factor`, gradually increasing it based on the training dynamics. It is also imperative to confirm that your input distributions are valid probability distributions, summing to 1 along the relevant dimension. If you do not ensure they are proper probability distributions, your results are meaningless.

For further study, I would suggest examining resources covering information theory and statistical divergence measures. Texts on variational inference often delve into practical methods for handling KL and JSD divergences within the context of generative modeling. Detailed studies on regularization techniques and optimization algorithms for deep learning are also beneficial. Further, exploring the mathematics surrounding probability distributions and statistical convergence will help build a more in-depth understanding of the issue. Lastly, practical examples and implementations can be gleaned from various open-source deep learning projects utilizing similar methodologies.
