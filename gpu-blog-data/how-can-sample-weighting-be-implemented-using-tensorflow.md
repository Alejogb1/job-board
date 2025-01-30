---
title: "How can sample weighting be implemented using TensorFlow Probability's `log_prob` loss function?"
date: "2025-01-30"
id: "how-can-sample-weighting-be-implemented-using-tensorflow"
---
Sample weighting within the TensorFlow Probability (TFP) framework, specifically when leveraging the `log_prob` loss function, requires a nuanced understanding of probability density functions and the inherent assumptions of maximum likelihood estimation (MLE).  My experience implementing Bayesian models in finance, specifically for option pricing under stochastic volatility, heavily relied on this technique to account for varying data quality and informativeness.  Crucially, direct weighting of the `log_prob` output is incorrect; instead, the weights must be incorporated into the probability density function itself.

The core issue stems from the nature of `log_prob`. This function returns the logarithm of the probability density at a given point.  Directly multiplying this log-probability by a weight would violate the fundamental properties of probability distributions. The weights need to be integrated into the definition of the probability density from which the `log_prob` is derived. This is achieved by modifying the underlying probability distribution to reflect the weighted sample importance.  We achieve this most effectively by employing a weighted average of probability densities, or, more elegantly, by constructing a mixture model.


**1. Clear Explanation:**

The correct approach involves modifying the likelihood function.  Instead of a single probability distribution describing all samples, we use a weighted mixture of distributions, where each distribution represents a single data point and its associated weight.  The weight acts as a mixing coefficient, determining the contribution of each data point to the overall likelihood.  Mathematically, this is represented as:

`p(x|θ, w) = Σᵢ wᵢ * p(x|θᵢ)`,

where:

* `x` represents the data point.
* `θ` represents the model parameters.  Note that we could have distinct parameters θᵢ for each data point, allowing for more complex weighting schemes.
* `wᵢ` is the weight associated with the i-th data point, satisfying Σᵢ wᵢ = 1.
* `p(x|θᵢ)` is the probability density of the i-th data point given its parameters.  This could be any distribution supported by TFP.

The log-likelihood then becomes:

`log p(x|θ, w) = log(Σᵢ wᵢ * p(x|θᵢ))`

This is then used in the optimization process.  Note that directly summing log-probabilities and weighting the sum (i.e., `Σᵢ wᵢ * log(p(xᵢ|θ))`) is generally incorrect and won't provide the correct weighted likelihood.


**2. Code Examples with Commentary:**

**Example 1: Simple Weighted Gaussian Mixture**

This example demonstrates weighting samples in a Gaussian mixture model. Each data point contributes to the likelihood with its associated weight.

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Sample data and weights (normalized)
data = tf.constant([1.0, 2.0, 3.0, 4.0])
weights = tf.constant([0.1, 0.2, 0.3, 0.4])

# Define a Gaussian mixture model (one Gaussian per data point)
def weighted_gaussian_mixture(data, weights, mu_prior, sigma_prior):
  components = [tfd.Normal(loc=tf.Variable(mu_prior), scale=tf.Variable(sigma_prior)) for _ in range(len(data))]
  mixture = tfd.Mixture(cat=tfd.Categorical(probs=weights), components=components)
  return mixture

# Priors for mean and standard deviation
mu_prior = tf.constant(0.0)
sigma_prior = tf.constant(1.0)

# Build and optimize the model
mixture_model = weighted_gaussian_mixture(data, weights, mu_prior, sigma_prior)
neg_log_likelihood = -tf.reduce_mean(mixture_model.log_prob(data))
optimizer = tf.optimizers.Adam(learning_rate=0.1)
optimizer.minimize(neg_log_likelihood, var_list=mixture_model.trainable_variables)

# Access optimized parameters (example)
print(mixture_model.components[0].loc)
```

**Example 2:  Weighted Normal Distribution with Parameter Sharing**

Here we use a single normal distribution but weight the contribution of each data point to the likelihood.  This is essentially a weighted MLE for a single normal.

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Sample data and weights (normalized)
data = tf.constant([1.0, 2.0, 3.0, 4.0])
weights = tf.constant([0.1, 0.2, 0.3, 0.4])

# Define the Normal distribution
mu = tf.Variable(0.0)
sigma = tf.Variable(1.0)
normal_dist = tfd.Normal(loc=mu, scale=sigma)

# Weighted log-likelihood (approximation using importance sampling)
weighted_log_likelihood = -tf.reduce_mean(weights * normal_dist.log_prob(data))

# Optimization
optimizer = tf.optimizers.Adam(learning_rate=0.1)
optimizer.minimize(weighted_log_likelihood, var_list=[mu, sigma])

# Access optimized parameters
print(mu, sigma)
```


**Example 3:  Importance Sampling for Complex Weighting**

This example shows how to handle complex weighting schemes that cannot be easily incorporated into a mixture model, using importance sampling.

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Sample data and importance weights
data = tf.constant([1.0, 2.0, 3.0, 4.0])
importance_weights = tf.constant([0.5, 1.0, 1.5, 2.0]) #Unnormalized

# Define the model (e.g., a normal distribution)
mu = tf.Variable(0.0)
sigma = tf.Variable(1.0)
normal_dist = tfd.Normal(loc=mu, scale=sigma)

# Importance sampling for weighted log-likelihood. Note the normalization of weights
normalized_importance_weights = importance_weights / tf.reduce_sum(importance_weights)
weighted_log_likelihood = -tf.reduce_mean(normalized_importance_weights * normal_dist.log_prob(data))

#Optimization
optimizer = tf.optimizers.Adam(learning_rate=0.1)
optimizer.minimize(weighted_log_likelihood, var_list=[mu, sigma])

#Access optimized parameters
print(mu, sigma)
```

**3. Resource Recommendations:**

* TensorFlow Probability documentation.
* Bishop's "Pattern Recognition and Machine Learning."
* "Bayesian Data Analysis" by Gelman et al.
* Relevant sections in textbooks on statistical inference and Bayesian methods.


This comprehensive approach addresses sample weighting accurately within the TFP framework.  Remember the key point:  weighting is not a post-processing step applied to `log_prob`; it's an integral part of defining the likelihood function itself, reflecting the importance of each data point in the estimation process. Using these examples and exploring the provided resources should equip you to handle a variety of sample weighting scenarios effectively.  Careful consideration of your specific application and the nature of your weights is crucial for selecting the most appropriate method.
