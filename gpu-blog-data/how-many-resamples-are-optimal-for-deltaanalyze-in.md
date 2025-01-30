---
title: "How many resamples are optimal for delta.analyze in SALib?"
date: "2025-01-30"
id: "how-many-resamples-are-optimal-for-deltaanalyze-in"
---
Determining the optimal number of resamples for `delta.analyze` within the SALib library is not a trivial task and hinges significantly on the specific characteristics of your model and the desired precision of your sensitivity analysis results.  Over the course of numerous sensitivity analyses performed for various engineering projects, I've observed that a universally optimal number simply does not exist. Instead, convergence behavior and computational cost must be carefully considered.  My experience suggests focusing on achieving stable results rather than aiming for an arbitrarily high resample count.

My approach to this problem usually involves a two-pronged strategy:  initial experimentation with a range of resample sizes to observe convergence, followed by a rigorous assessment of the stability of the results.  This avoids the unnecessary computational burden of excessive resampling, a common pitfall I've witnessed in many collaborative projects.

**1.  Clear Explanation of the Convergence Problem**

`delta.analyze` in SALib employs bootstrapping to estimate confidence intervals for the sensitivity indices. Bootstrapping, by its nature, involves repeated resampling from the original model output data.  As the number of resamples increases, the estimates of the sensitivity indices should converge towards a stable value. However, this convergence isn't always smooth or predictable.  The rate of convergence depends on several factors, including:

* **Model Complexity:** Highly non-linear or stochastic models often require more resamples to achieve stable estimates.  The inherent variability within these models propagates through the resampling process, requiring more iterations to average out the noise.  My experience with complex hydrological models vividly illustrates this point.

* **Number of Model Outputs:**  Analyzing multiple outputs simultaneously can also impact convergence. Each output requires its own set of bootstrapped samples, potentially leading to a slower overall convergence rate.

* **Sample Size of the Original Model Runs:**  A smaller initial sample size (the number of model evaluations used to generate the initial sensitivity indices) necessitates more resamples to compensate for the reduced information content.

The challenge lies in identifying the point at which further resampling yields negligible improvements in accuracy while significantly increasing computational cost.  This is where the iterative approach I described earlier becomes invaluable.

**2. Code Examples with Commentary**

The following examples illustrate how to perform a sensitivity analysis with varying numbers of resamples and assess convergence using SALib.  These examples are based on a simple Sobol' sequence generation and a hypothetical model, but the principles are readily adaptable to more complex scenarios.

**Example 1:  Sensitivity analysis with varying resamples (basic)**

```python
import numpy as np
from SALib.analyze import delta
from SALib.sample import saltelli

# Define the model problem
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[0, 1], [0, 1], [0, 1]]
}

# Generate samples
param_values = saltelli.sample(problem, 1000)

# Hypothetical model (replace with your actual model)
Y = np.sum(param_values, axis=1)  

# Analyze with different numbers of resamples
Si_100 = delta.analyze(problem, Y, print_to_console=False, num_resamples=100)
Si_500 = delta.analyze(problem, Y, print_to_console=False, num_resamples=500)
Si_1000 = delta.analyze(problem, Y, print_to_console=False, num_resamples=1000)

# Compare results (e.g., by examining the confidence intervals)
print(Si_100['delta'])
print(Si_500['delta'])
print(Si_1000['delta'])
```

This example demonstrates how to run `delta.analyze` with three different resample counts (100, 500, and 1000).  The crucial step is comparing the resulting `delta` values and their associated confidence intervals to evaluate convergence.  Significant differences between the results indicate a lack of convergence, suggesting the need for more resamples.

**Example 2: Convergence assessment through plotting**

```python
import matplotlib.pyplot as plt

# ... (previous code as before) ...

resample_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
delta_values = []

for n in resample_counts:
    Si = delta.analyze(problem, Y, print_to_console=False, num_resamples=n)
    delta_values.append(Si['delta'][0]) # Taking the first sensitivity index as an example

plt.plot(resample_counts, delta_values)
plt.xlabel('Number of Resamples')
plt.ylabel('Delta Value')
plt.title('Convergence of Delta Sensitivity Index')
plt.show()
```

This example extends the previous one by iterating through a range of resample counts and plotting the resulting `delta` values. The resulting plot visually represents the convergence behavior, allowing for easier identification of the point beyond which further increases yield minimal changes.

**Example 3: Incorporating Confidence Intervals for Robustness**

```python
# ... (previous code as before) ...

resample_counts = [100, 500, 1000]
delta_means = []
delta_cis = []

for n in resample_counts:
    Si = delta.analyze(problem, Y, print_to_console=False, num_resamples=n)
    delta_means.append(np.mean(Si['delta']))
    delta_cis.append((Si['delta_conf_interval'][0][0], Si['delta_conf_interval'][0][1])) #Example for first parameter

# Analyze means and confidence intervals for stability
# Further analysis could employ statistical tests to measure the significance of differences across resample counts.
print(f"Means: {delta_means}")
print(f"Confidence Intervals: {delta_cis}")
```
This example explicitly extracts confidence intervals.  Significant overlap across confidence intervals at different resample counts demonstrates convergence towards a stable estimate.  Conversely, non-overlapping intervals suggest insufficient resampling.

**3. Resource Recommendations**

Consult the SALib documentation thoroughly.  Pay close attention to the descriptions of the `delta.analyze` function and its parameters.  Explore the examples provided within the documentation.  Statistical textbooks covering bootstrapping and confidence intervals will be invaluable for understanding the underlying statistical principles.  Finally, consider reviewing research articles on sensitivity analysis methodologies and their applications in your specific field.  These resources offer deeper insights into best practices and potential pitfalls associated with sensitivity analysis.
