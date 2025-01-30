---
title: "How do I calculate probability density for a given distribution in PyTorch?"
date: "2025-01-30"
id: "how-do-i-calculate-probability-density-for-a"
---
Probability density estimation is frequently encountered in my work on Bayesian neural networks, particularly when dealing with latent variable models and variational inference.  A critical understanding lies in distinguishing between probability mass functions (PMFs) for discrete variables and probability density functions (PDFs) for continuous variables.  PyTorch doesn't directly offer a single function to compute the *density* of an arbitrary distribution; instead, it provides tools to work with distributions and sample from them, allowing us to approximate the density. The approach depends heavily on the type of distribution.

1. **Direct Density Calculation (for known distributions):**  If you're working with a standard distribution implemented in `torch.distributions`, like Normal, Gamma, or Beta, you can directly evaluate the probability density function.  These distributions have a built-in `log_prob` method, returning the natural logarithm of the probability density. This is generally preferred for numerical stability, especially when dealing with small probabilities.  Exponentiating the result yields the density.

   ```python
   import torch
   from torch.distributions import Normal

   # Define a normal distribution
   mu = torch.tensor([0.0])
   sigma = torch.tensor([1.0])
   normal_dist = Normal(mu, sigma)

   # Evaluate the density at a specific point
   x = torch.tensor([1.0])
   log_prob_x = normal_dist.log_prob(x)
   prob_density_x = torch.exp(log_prob_x)

   print(f"Log probability density at x=1.0: {log_prob_x.item()}")
   print(f"Probability density at x=1.0: {prob_density_x.item()}")


   # Example with a batch of inputs
   x_batch = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
   log_prob_batch = normal_dist.log_prob(x_batch)
   prob_density_batch = torch.exp(log_prob_batch)
   print(f"Log Probability density for batch: \n{log_prob_batch}")
   print(f"Probability density for batch: \n{prob_density_batch}")
   ```

   This code demonstrates the straightforward approach for known distributions. The `log_prob` method efficiently handles both single points and batches of points.  Remember, for multi-dimensional distributions, the input `x` should have the appropriate shape.


2. **Kernel Density Estimation (KDE) for unknown distributions:** When dealing with an empirical distribution or a distribution you cannot easily parameterize, KDE is a powerful non-parametric technique.  KDE estimates the PDF by placing a kernel (e.g., Gaussian) at each data point and summing the kernels. The bandwidth of the kernel is a crucial hyperparameter affecting the smoothness of the estimated density.  I've found the `scikit-learn` library, despite not being directly PyTorch, to be exceptionally useful for this.  PyTorch tensors can be seamlessly integrated.


   ```python
   import torch
   import numpy as np
   from sklearn.neighbors import KernelDensity

   # Sample data (replace with your data)
   data = torch.randn(100, 1).numpy() #Using numpy for sklearn compatibility

   # Create and fit the KDE model
   kde = KernelDensity(bandwidth=0.5, kernel='gaussian') #Adjust bandwidth as needed
   kde.fit(data)

   # Evaluate the density at specific points
   x_eval = np.linspace(-3, 3, 100)[:, np.newaxis]
   log_density = kde.score_samples(x_eval)
   density = np.exp(log_density)


   #Convert back to pytorch tensor for further use in PyTorch models.
   density_pt = torch.tensor(density)
   x_eval_pt = torch.tensor(x_eval)

   print(f"Density estimates:\n {density_pt}")

   ```

   Note that this code utilizes `numpy` for compatibility with `scikit-learn`.  The result, however, is easily converted back to a PyTorch tensor for further processing within your PyTorch workflow. The bandwidth parameter significantly influences the results; experimentation and cross-validation are often necessary for optimal results.


3. **Histogram-based Approximation:**  A simpler, albeit less accurate, method is to approximate the density using a histogram. This is particularly useful for visualizing the density.  By normalizing the histogram counts, you obtain an approximation of the probability density.


   ```python
   import torch
   import matplotlib.pyplot as plt

   # Sample data (replace with your data)
   data = torch.randn(1000)

   # Create a histogram
   hist, bin_edges = torch.histc(data, bins=20)

   # Normalize the histogram to get an approximation of the density
   bin_width = bin_edges[1] - bin_edges[0]
   density_approx = hist / (torch.sum(hist) * bin_width)

   #Plot the histogram for visualization.
   plt.hist(data, bins=20, density=True)
   plt.plot(bin_edges[:-1] + bin_width/2, density_approx, 'r-', label='Density Approximation')
   plt.xlabel("Value")
   plt.ylabel("Density")
   plt.legend()
   plt.show()

   ```

   This approach offers a quick and intuitive way to visualize the density.  The accuracy, however, is limited by the bin size and the number of bins used.  For precise density estimation, KDE or direct calculations for known distributions are preferred.


**Resource Recommendations:**

For a thorough understanding of probability distributions and density estimation, I strongly suggest consulting standard textbooks on probability and statistics.  Furthermore, the official PyTorch documentation provides comprehensive details on the `torch.distributions` module and its various functions.  Finally, a good book on numerical methods will provide insight into the various techniques used for approximating densities.  Exploring these resources will provide a strong foundation to tackle more complex probabilistic modeling tasks.
