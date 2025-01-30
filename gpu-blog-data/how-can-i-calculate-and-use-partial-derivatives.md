---
title: "How can I calculate and use partial derivatives of probability functions in PyTorch?"
date: "2025-01-30"
id: "how-can-i-calculate-and-use-partial-derivatives"
---
Automatic differentiation, central to PyTorch’s functionality, allows the computation of gradients for arbitrary, differentiable operations, including those involved in probability distributions. This capability is crucial for optimizing parameters in probabilistic models, such as neural networks that output parameters of distributions. Utilizing partial derivatives of probability functions in PyTorch, therefore, typically involves representing the probability function using PyTorch tensor operations, setting requires_grad=True for relevant parameters, and then invoking the .backward() method on a scalar value derived from the probability.

The core principle is that PyTorch automatically tracks operations performed on tensors with `requires_grad=True`. When `.backward()` is called on a scalar, it propagates gradients backward through the computation graph, effectively calculating partial derivatives with respect to these tracked tensors. For probability functions, we usually work with quantities derived from them, like the negative log-likelihood (NLL), rather than directly differentiating the probability function itself. The NLL is particularly useful because minimizing it maximizes the likelihood of observed data under the assumed model.

Let's examine how this works concretely with a few examples.

**Example 1: Derivatives of a Gaussian Distribution**

Suppose we want to train a simple model that predicts the mean and standard deviation of a Gaussian distribution, and then optimize these parameters based on observed data. The Gaussian probability density function is defined as:

```
f(x; μ, σ) = 1 / (σ * sqrt(2π)) * exp(-(x - μ)^2 / (2σ^2))
```

However, it is more numerically stable to work with the negative log-likelihood (NLL) of the distribution.  The NLL for a single observation x is:

```
NLL(x; μ, σ) = 0.5 * log(2π) + log(σ) + (x - μ)^2 / (2σ^2)
```

Here's how you would implement this in PyTorch and calculate partial derivatives:

```python
import torch
import torch.distributions as dist

# Observed Data
observed_data = torch.tensor([2.0])

# Initialize Mean and Standard Deviation parameters, requires_grad = True
mu = torch.tensor([0.0], requires_grad=True)
sigma = torch.tensor([1.0], requires_grad=True)

# Create a normal distribution with the parameters
normal_dist = dist.Normal(mu, sigma)

# Calculate the negative log likelihood (NLL) of the data given the distribution
nll = -normal_dist.log_prob(observed_data).mean()

# Perform backpropagation to compute the gradients of NLL wrt mu and sigma
nll.backward()

# Print the gradients of the NLL wrt mu and sigma
print(f"Gradient of NLL with respect to mu: {mu.grad}")
print(f"Gradient of NLL with respect to sigma: {sigma.grad}")

# Example of how to use the gradients to update parameters.
learning_rate = 0.1
with torch.no_grad():
    mu -= learning_rate * mu.grad
    sigma -= learning_rate * sigma.grad
    mu.grad.zero_()
    sigma.grad.zero_()
print(f"Updated mu: {mu}")
print(f"Updated sigma: {sigma}")
```

**Commentary:**
   1. We define our observed data and our parameters as PyTorch tensors, with `requires_grad=True` for the parameters.
   2. We create a `torch.distributions.Normal` object. This internally represents the Gaussian distribution with the given parameters.
   3. We use `.log_prob()` to obtain log probabilities and the `.mean()` over possible samples to account for distributions where multiple samples may be present at any given time. In this case, `.mean()` doesn't actually affect the loss, but is good practice for batched inputs. The negative sign in front results in the NLL.
   4.  The `nll.backward()` call computes the gradients of the NLL with respect to `mu` and `sigma`.
   5. We access the gradients using the `.grad` property on `mu` and `sigma`.
   6. Finally, we demonstrate how these gradients can be used to update the distribution parameters. Note the `torch.no_grad()` context, required when updating, and `zero_()` call to clear the old gradients.

**Example 2: Derivatives with respect to probabilities of a Categorical Distribution**

Consider a scenario where you model an outcome as arising from a categorical distribution, and your model predicts the *probabilities* associated with each category. In this case, we'll need to work with the probability values directly (though they are typically outputs of a softmax activation) and apply the negative log-likelihood.

```python
import torch
import torch.distributions as dist
import torch.nn.functional as F

# Observed category index
observed_category = torch.tensor([2])

# Predicted logits (pre-softmax) for each category, requires_grad = True
logits = torch.tensor([0.1, 0.5, 1.2], requires_grad=True)

# Apply softmax to get the probabilities
probabilities = F.softmax(logits, dim=-1)

# Create a categorical distribution with the probabilities
categorical_dist = dist.Categorical(probs=probabilities)

# Calculate the negative log likelihood of the observed data
nll = -categorical_dist.log_prob(observed_category)

# Perform backpropagation
nll.backward()

# Print the gradients of the NLL wrt logits
print(f"Gradients with respect to logits: {logits.grad}")
```

**Commentary:**
1. The `logits` tensor represents the raw outputs from a model that predict probabilities, before the softmax is applied.
2. `F.softmax` converts the logits to probability values.
3. The distribution is constructed using these probability values.
4. The remainder is very similar to Example 1. We compute the negative log-likelihood of the observed category, call `backward()`, and inspect the gradients. Note that the gradients are computed with respect to the *logits* not the probabilities themselves.
5. Backpropagation tracks the `softmax` operation as well, allowing for optimization of parameters that influence the `logits`.

**Example 3:  Using a Custom Probability Distribution**

While PyTorch provides built-in distribution classes, you may need to define custom distributions for specific problems. Here's how you could calculate derivatives for a custom "triangular" distribution:

```python
import torch
import torch.distributions as dist

class Triangular(dist.Distribution):
    def __init__(self, lower, peak, upper):
        super().__init__()
        self.lower = lower
        self.peak = peak
        self.upper = upper
        if not (lower <= peak <= upper):
            raise ValueError("Must be lower <= peak <= upper")

    def log_prob(self, value):
        if (value < self.lower or value > self.upper):
           return -torch.inf
        elif value < self.peak:
          return torch.log(2*(value - self.lower)/((self.peak-self.lower)*(self.upper-self.lower)))
        else:
          return torch.log(2*(self.upper - value)/((self.upper-self.peak)*(self.upper-self.lower)))

# Observed data
observed_data = torch.tensor([3.0])

# Define parameters, needs_grad=True
lower_bound = torch.tensor([1.0], requires_grad=True)
peak_value = torch.tensor([3.5], requires_grad=True)
upper_bound = torch.tensor([5.0], requires_grad=True)

# Create triangular distribution
triangular_dist = Triangular(lower_bound, peak_value, upper_bound)

# Calculate NLL
nll = -triangular_dist.log_prob(observed_data).mean()

# Backpropagation
nll.backward()

# Print the gradients
print(f"Gradient wrt lower bound: {lower_bound.grad}")
print(f"Gradient wrt peak: {peak_value.grad}")
print(f"Gradient wrt upper bound: {upper_bound.grad}")
```

**Commentary:**
1. Here, we create a class `Triangular` to represent the distribution, inherit from `torch.distributions.Distribution`, and define our `log_prob` function according to its probability density.
2. The implementation of the custom `log_prob()` function must be numerically stable, and should be differentiable with respect to the input parameters if we expect to optimize the parameters of this custom distribution. It is important to also use `torch` operations to achieve this.
3.  The rest of the process is identical to previous examples. We compute the NLL, call `backward()`, and access gradients through the `.grad` attribute.

**Resource Recommendations**

To deepen understanding, consult the official PyTorch documentation on:

*   Autograd (for automatic differentiation)
*   torch.distributions (for built-in probability distributions)
*   torch.nn.functional (for activation functions like softmax)

Additionally, introductory material on Bayesian inference and maximum likelihood estimation would prove helpful for framing these computations in a larger modeling context.  Furthermore, reviewing examples of variational autoencoders or other neural network based generative models is recommended for practical application. Books focusing on probabilistic machine learning also offer a good overview of how these gradient-based calculations are applied.
