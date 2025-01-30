---
title: "How can differential privacy be implemented in federated learning?"
date: "2025-01-30"
id: "how-can-differential-privacy-be-implemented-in-federated"
---
Differential privacy (DP) provides a rigorous framework for protecting individual privacy when analyzing aggregate datasets. Applying DP to federated learning (FL), where models are trained across decentralized data sources, presents unique challenges due to the iterative nature of model aggregation and the need to preserve the privacy of participating clients’ data. I have spent a significant amount of time exploring methods to integrate these two potent techniques, and I will detail my experience with implementing them together.

The core issue in achieving differential privacy in FL is that, while each client trains a model locally using their private data, the updates or parameters of these models are then shared with a central server to produce a globally aggregated model. This sharing process has the potential to reveal information about individual client data, therefore violating the principles of DP. Directly applying DP mechanisms to the model parameters at each client would likely result in a severely degraded global model due to the compounding noise. It is vital that the noise be introduced carefully, and that its effect be appropriately managed.

The approach that I have found most effective centers around the application of DP at the client level *before* sending updates to the central server. Specifically, this involves clipping the gradients, adding noise before sending, and controlling the total privacy budget. The process, in essence, becomes: 1) Client computes gradients on local data, 2) gradients are clipped to prevent individual large updates from dominating, 3) controlled noise is added to the clipped gradients, and 4) noised gradients are shared with the server for aggregation. The key aspects are clipping the individual gradients, as well as determining appropriate sensitivity and noise levels.

Let's explore the implementation via some Python-based illustrative examples, demonstrating some essential steps. The following are conceptual, they have some dependencies abstracted and are not meant to be run directly.

**Example 1: Gradient Clipping**

Here, I demonstrate clipping the gradients computed during local model training. It is important to perform clipping before adding noise, because the scale of noise is tied to sensitivity, and unbounded gradients will have unbounded sensitivities. Let's assume a simple model training process where each client computes gradients using stochastic gradient descent (SGD).

```python
import numpy as np

def clip_gradients(gradients, clip_norm):
    """
    Clips the gradients using L2 norm clipping.

    Args:
        gradients: A list or numpy array of gradients.
        clip_norm: The maximum allowed L2 norm for gradients.

    Returns:
        Clipped gradients.
    """
    gradient_norm = np.linalg.norm(np.concatenate([g.flatten() for g in gradients])) # Flatten and compute norm
    if gradient_norm > clip_norm:
        clip_coef = clip_norm / gradient_norm
        clipped_gradients = [g * clip_coef for g in gradients]
    else:
        clipped_gradients = gradients
    return clipped_gradients

# Example Usage (assuming a list of gradient arrays)
gradients = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
clip_norm = 5
clipped_gradients = clip_gradients(gradients, clip_norm)

print(f"Original gradients: {gradients}")
print(f"Clipped gradients: {clipped_gradients}")
```

In this example, `clip_gradients` takes a list of gradient arrays and a `clip_norm` value. It calculates the L2 norm of the combined gradients. If the norm exceeds the `clip_norm`, all gradients are scaled down such that their combined norm exactly matches `clip_norm`. This ensures that no individual client’s update can unduly influence the aggregated model. This is a critical step for bounding the sensitivity of the computation prior to noise addition. If clipping is not applied, a large gradient from one client might obscure privacy in spite of the introduced noise.

**Example 2: Adding Laplacian Noise**

Next, I show how to add Laplacian noise to the clipped gradients. I favor the Laplacian distribution because it is commonly used in DP. The scale of the Laplacian noise is determined by the sensitivity of the gradient aggregation process and the desired privacy parameter, epsilon.

```python
def add_laplacian_noise(gradients, sensitivity, epsilon):
    """
    Adds Laplacian noise to gradients for differential privacy.

    Args:
        gradients: The clipped gradients.
        sensitivity: The sensitivity of the gradient computation.
        epsilon: The privacy parameter.

    Returns:
        Gradients with added noise.
    """
    laplace_scale = sensitivity / epsilon
    noised_gradients = [g + np.random.laplace(0, laplace_scale, g.shape) for g in gradients]
    return noised_gradients

# Example Usage
sensitivity = 2  # Sensitivity based on clipping and maximum change from one data point
epsilon = 1 # Lower epsilon for greater privacy.
noised_gradients = add_laplacian_noise(clipped_gradients, sensitivity, epsilon)

print(f"Clipped gradients: {clipped_gradients}")
print(f"Noised gradients: {noised_gradients}")

```

The function `add_laplacian_noise` calculates the scale of the Laplacian noise based on the given sensitivity and epsilon. The sensitivity represents the maximum change in gradients caused by an individual record and is directly related to the clipping norm (since each gradient is clipped). The noise is added to each of the gradient components. The choice of epsilon and sensitivity is a balancing act. Lower values of epsilon increase privacy but require larger amounts of noise, and can reduce the model's performance. Sensitivity, which is tied to the choice of clipping value, should be carefully evaluated and tuned.

**Example 3: Privacy Budget Tracking**

Finally, privacy budget accounting is essential. Each time gradients are sent to the server, a portion of the total privacy budget (ε) is consumed. It is vital to track this consumed budget, since we will want to tune parameters and understand the total privacy loss in a given training run.

```python
class PrivacyAccountant:
    """
    Tracks privacy budget usage in a federated learning system.
    """
    def __init__(self, total_epsilon):
        self.total_epsilon = total_epsilon
        self.epsilon_spent = 0

    def spend_epsilon(self, current_epsilon):
      """
      Updates the total spent epsilon.
      Args:
        current_epsilon: epsilon value of current computation.
      """
      self.epsilon_spent += current_epsilon

    def get_remaining_epsilon(self):
      """
      Returns the remaining epsilon.
      """
      return self.total_epsilon - self.epsilon_spent

    def check_budget(self):
      """
      Checks if the budget is exceeded.
      """
      if self.epsilon_spent > self.total_epsilon:
        print("Warning: privacy budget exceeded!")

# Example Usage
total_epsilon = 5  # Set total allowed epsilon.
privacy_tracker = PrivacyAccountant(total_epsilon)

# Example in a training loop, assuming `noised_gradients` has been produced
epsilon_per_round = 1
privacy_tracker.spend_epsilon(epsilon_per_round)

print(f"Epsilon spent: {privacy_tracker.epsilon_spent}")
print(f"Remaining epsilon: {privacy_tracker.get_remaining_epsilon()}")
privacy_tracker.check_budget()

privacy_tracker.spend_epsilon(7)
privacy_tracker.check_budget()
```

The `PrivacyAccountant` class maintains the total allowed epsilon and tracks how much has been spent, or consumed. This lets me monitor privacy loss and ensures that the total privacy budget is not exceeded. The spent budget depends on the number of clients, number of iterations, and the sensitivity of each step. In a more comprehensive implementation, a privacy budget management scheme is more likely to be based on the RDP framework, especially when Gaussian noise is being used, but this provides a good starting point. This class can be extended to use more sophisticated accounting methods, particularly with Gaussian noise, but this gives an overview of the needed principles.

Implementing DP in FL requires careful consideration of several factors, including the choice of noise mechanism (Laplacian, Gaussian, etc.), sensitivity, epsilon (privacy parameter), and delta (probability of failing to maintain DP guarantee). I have found that clipping plays a crucial role as a sensitivity bounder, and therefore should not be overlooked. These code examples, while simplified, demonstrate the critical steps involved in adding DP to federated learning. They form the basis of a functional, differentially private training process.

For further learning on this topic I would recommend the following resources: “The Algorithmic Foundations of Differential Privacy” by Dwork and Roth provides a theoretical understanding. “Deep Learning with Differential Privacy” by Abadi et al. is invaluable for its coverage of practical implementation. There are numerous research papers on federated learning with differential privacy. I found it useful to explore those which focus on the practical limitations and how they address them. Lastly, I suggest looking into publicly available implementations. Some libraries provide frameworks for differential privacy which can be adapted to a federated setting, but remember that a deep understanding of principles is needed to make the best adjustments for your specific use case.
