---
title: "How can automatic differentiation be applied to rank-based computations?"
date: "2025-01-30"
id: "how-can-automatic-differentiation-be-applied-to-rank-based"
---
Automatic differentiation (AD) significantly simplifies the implementation and optimization of complex machine learning models, including those employing rank-based metrics. Historically, optimizing rank-based losses has been challenging due to their non-differentiability. However, AD techniques, coupled with appropriate relaxations, enable us to overcome this barrier. My experience building search relevance systems highlighted the practical need for these approaches.

**1. The Challenge of Rank-Based Metrics**

Rank-based metrics, like Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), or recall at k, are ubiquitous in information retrieval and recommendation systems. These metrics measure the quality of a ranked list of items. Unlike loss functions operating on individual data points, these metrics assess the order or ranking produced by the model. A critical issue is that direct calculation of these metrics involves discrete operations such as sorting and indexing, rendering them non-differentiable. Backpropagation, a core mechanism in training neural networks, requires differentiable functions to calculate gradients, thus presenting a hurdle.

Traditional approaches bypassed this issue by employing techniques like policy gradients or evolutionary algorithms, which do not require gradients of the loss function. However, these methods typically suffer from high variance and computational expense. Furthermore, they often require careful hyperparameter tuning, making them less practical in many cases. Therefore, directly differentiable methods that can leverage the power of AD are preferable.

**2. Relaxations for Differentiable Ranking**

To make rank-based metrics amenable to AD, we employ relaxation techniques. These techniques involve replacing the non-differentiable sorting or ranking operations with smooth approximations. Key concepts underpinning these relaxations include:

*   **Soft Ranking:** Instead of assigning a definite rank to each item, we assign soft ranks which are continuous and differentiable. This is often achieved through functions based on sigmoids or similar smooth functions. The intuition is that instead of a hard sorting, each item gets a "probability" of being higher or lower ranked compared to others.
*   **Smooth Approximations of Step Function:** Rank metrics like NDCG are based on indicators whether an item is relevant or not, which are step functions (0/1). These can be approximated by smooth, sigmoid-like curves. For example, instead of saying an item is definitely relevant or irrelevant (1 or 0), we use a function that varies continuously between 0 and 1, reflecting the degree of relevance.
*   **Plackett-Luce Model:** This probabilistic model is widely used in ranking. It provides the probability distribution over possible permutations of rankings based on scores assigned to the items. If scores are derived from differentiable models, probabilities over rankings and expectations can become differentiable too.

By implementing these relaxations, we enable the propagation of gradients through the metric calculation, allowing us to train the underlying model using gradient-based methods like stochastic gradient descent (SGD).

**3. Code Examples with Commentary**

Here, I'll demonstrate three simplified code examples focusing on core differentiable ranking components. These examples are in Python using `torch` as the AD library.

**Example 1: Soft Rank Approximation**

This example shows how to compute soft ranks.

```python
import torch
import torch.nn.functional as F

def soft_rank(scores):
    """Computes soft ranks from scores."""
    scores_exp = torch.exp(scores)
    return 1.0 + torch.sum(scores_exp / (scores_exp.unsqueeze(1) + 1e-8), dim=2)

# Example usage:
scores = torch.tensor([[[1.0, 2.0, 0.5], [0.2, 0.8, 1.5]], [[1.2, 0.7, 0.9], [1.6, 1.3, 1.1]]], requires_grad=True)

soft_ranks = soft_rank(scores)
print("Soft Ranks:\n", soft_ranks)
```

**Commentary:**

*   The `soft_rank` function takes a tensor of scores.
*   It calculates `exp(scores)` for each score which is used to introduce continuous approximation.
*   The main approximation is in the `torch.sum()` portion. For each score, it calculates sum of how many other scores are smaller in the dataset and creates a smooth rank.
*   The `unsqueeze` adds a dimension, allowing element-wise comparison between scores. The `1e-8` is an epsilon term for numerical stability.
*   The resulting `soft_ranks` tensor contains approximated rank values that are differentiable with respect to input scores.

**Example 2: Differentiable NDCG Approximation**

This example showcases the computation of a simplified differentiable approximation of NDCG.

```python
def differentiable_ndcg(scores, labels):
    """Computes differentiable NDCG approximation."""
    sorted_labels = labels[torch.argsort(scores, dim=-1, descending=True)]
    gains = (2 ** sorted_labels - 1).float()
    discount = torch.log2(torch.arange(2, scores.size(-1) + 2).float())
    dcg = torch.sum(gains / discount, dim=-1)

    # IDCG: ideal DCG based on sorted labels
    ideal_sorted_labels = torch.sort(labels, dim=-1, descending=True)[0]
    ideal_gains = (2 ** ideal_sorted_labels - 1).float()
    idcg = torch.sum(ideal_gains / discount, dim=-1)

    return dcg / idcg
    # return torch.mean(dcg / idcg)

# Example Usage
scores = torch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 1.5]], requires_grad=True)
labels = torch.tensor([[1, 0, 2], [0, 1, 1]]).float()
ndcg_approx = differentiable_ndcg(scores, labels)
print("Differentiable NDCG:\n", ndcg_approx)

# Loss: minimize negative differentiable ndcg to maximize the original NDCG
loss = - torch.mean(ndcg_approx)
loss.backward()
print("Gradients:\n", scores.grad)

```

**Commentary:**

* The `differentiable_ndcg` function calculates NDCG using differentiable approximation.
* The `torch.argsort` is not strictly differentiable, but it produces a gradient that allows optimization to continue.
* Discount is calculated using the log2 function, which is also differentiable.
*   We then compute both DCG and the ideal DCG using sorted labels, then divide the DCG by IDCG.
*   The loss is calculated as the negative average of the differentiable NDCG which we minimize.
*   Calling `loss.backward()` computes gradients with respect to `scores`.

**Example 3: Using Plackett-Luce for Ranking Probabilities**

This example applies the Plackett-Luce model to compute differentiable ranking probabilities, using `torch.distributions` which is specifically meant to produce differentiable distributions.

```python
import torch
from torch.distributions import Categorical

def plackett_luce(scores):
    """Computes probabilities of different permutations using Plackett-Luce."""
    scores_exp = torch.exp(scores)

    # Create a tensor of indices to iterate over
    num_items = scores.size(-1)
    indices = torch.arange(num_items)

    # Initialize probabilities
    probabilities = torch.ones(scores.shape[:-1], dtype=scores.dtype, device=scores.device)

    for i in range(num_items):
      remaining_scores = scores_exp[:, i:] # remaining scores
      scores_for_current_item = remaining_scores[:,0]
      sum_of_remaining_scores = torch.sum(remaining_scores, dim=-1)

      # probability of the current choice at its turn
      probability_i = scores_for_current_item / sum_of_remaining_scores

      probabilities = probabilities * probability_i
    return probabilities

# Example usage
scores = torch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 1.5]], requires_grad=True)
ranking_probabilities = plackett_luce(scores)
print("Plackett-Luce Probabilities:\n", ranking_probabilities)

# Calculate expected rank position (simplified)
expected_ranks = torch.zeros_like(scores)
for i in range(scores.size(-1)):
    expected_rank_i = torch.sum(ranking_probabilities * torch.arange(scores.size(-1), dtype=scores.dtype, device=scores.device))
    expected_ranks[:,i] = expected_rank_i

print("Expected Ranks:\n", expected_ranks)

# Loss function : difference between target rank and expected rank.
target_ranks = torch.tensor([[0,1,2],[2,1,0]], dtype=torch.float, device=scores.device)

loss = torch.mean(torch.abs(target_ranks - expected_ranks))
loss.backward()
print("Gradients:\n", scores.grad)
```

**Commentary:**

*   The `plackett_luce` function returns the probabilities of each possible ranking permutation.
*   We iterate through the items calculating the probability of each being the next one in the ranking.
*   Expected rank is calculated as the sum of ranks weighted by their respective probabilities.
*   The loss function is calculated by the difference between the target ranks and the expected ranks.
*   The gradients are calculated for `scores`.

**4. Resource Recommendations**

For further exploration, consider texts focusing on machine learning optimization, probabilistic modeling, and information retrieval. Specific texts on reinforcement learning can shed light on alternative approaches. Additionally, academic papers focused on "learning to rank" offer in-depth treatments of these topics. Online courses on deep learning frameworks, such as PyTorch or TensorFlow, can provide practical knowledge for implementing these concepts.
