---
title: "How do I compute power analysis with the Wilcoxon signed-rank test?"
date: "2024-12-16"
id: "how-do-i-compute-power-analysis-with-the-wilcoxon-signed-rank-test"
---

Alright, let's tackle this. Power analysis, especially with non-parametric tests like the Wilcoxon signed-rank, can seem a bit labyrinthine at first, but it's absolutely crucial for robust research design. I've certainly seen projects go sideways because this step was skipped or mishandled. I recall one particular project involving sensor data from wearable devices; a perfectly valid hypothesis, but the study was drastically underpowered and resulted in a lot of wasted effort. So, let's break down how to approach this for the Wilcoxon test, keeping in mind it isn’t quite as straightforward as with t-tests.

The core idea behind power analysis is to determine the sample size needed to detect a statistically significant effect with a specified probability (the power), given a particular effect size, alpha level, and the characteristics of your test. With the Wilcoxon signed-rank test, we’re dealing with ranks, not means and standard deviations, so the calculations become a bit less direct. A vital first consideration here is that the Wilcoxon test does not directly lend itself to classic power calculations that use effect sizes in terms of mean differences. Instead, we will deal with *pseudo-effect sizes* that capture the degree of shift between two distributions.

The first thing to acknowledge is that closed-form solutions for power analysis of the Wilcoxon test are rare. This is where simulation studies become your best friend. I have utilized them many times in the past and will again here. The basic idea is to simulate data according to the null and alternative hypotheses, then run the Wilcoxon test and see how frequently the alternative hypothesis is correctly recognized at a given sample size and effect size. The main challenge lies in how you define the effect size, since the Wilcoxon test is fundamentally about rank differences rather than raw differences.

We can use a probability shift parameter, sometimes denoted as *p*, to characterize the effect size. In the null hypothesis, we assume that differences between paired observations are symmetrically distributed around zero, resulting in *p* = 0.5. The probability shift *p* then represents the probability that the median of differences is greater than 0 when the null hypothesis of no difference is false. A value of *p* greater than 0.5 indicates the alternative hypothesis is true, and its magnitude represents the strength of this effect. For example, *p* = 0.7 indicates that the probability of a positive difference is much larger than the probability of a negative difference.

Here's how a typical simulation setup might look using Python. We'll use scipy and numpy to generate random samples and carry out the statistical tests.

```python
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

def simulate_wilcoxon_power(n, p_alt, num_sims=1000, alpha=0.05):
    """
    Simulates power for the Wilcoxon signed-rank test.

    Args:
      n: sample size (number of pairs)
      p_alt: The probability of a positive difference under the alternative.
      num_sims: Number of simulations to perform.
      alpha: Significance level.

    Returns:
        float: estimated power (proportion of simulations that correctly reject the null).
    """

    reject_count = 0
    for _ in range(num_sims):
       # Sample paired differences with probability `p_alt` for a positive difference.
        differences = np.random.choice([-1, 1], size=n, p=[1-p_alt, p_alt])

        # Run the wilcoxon signed-rank test
        stat, p_val = wilcoxon(differences, alternative='greater')

        if p_val < alpha:
            reject_count += 1
    return reject_count / num_sims

# Example usage:
sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80]
powers = []

for n in sample_sizes:
    power_est = simulate_wilcoxon_power(n, p_alt=0.7, num_sims=500)
    powers.append(power_est)
    print(f"Sample size: {n}, Power: {power_est}")

plt.plot(sample_sizes, powers)
plt.xlabel('Sample size')
plt.ylabel('Power')
plt.title('Power curve for wilcoxon test at p_alt = 0.7')
plt.show()
```

In this snippet, `simulate_wilcoxon_power` performs the simulation, and the example shows how you would iterate through multiple sample sizes to observe the increase in power. The p_alt represents the probability shift discussed above. The plot at the end will help visualize how power increases with the sample size given a fixed p_alt of 0.7.

Now, let's look at a variation using a more practical data simulation. We’ll simulate the paired data using two separate distributions and calculate the difference between each pair. This is useful when you have an expectation or prior data on the distribution you are working with:

```python
import numpy as np
from scipy.stats import wilcoxon, norm
import matplotlib.pyplot as plt

def simulate_wilcoxon_power_distributional(n, shift, num_sims=1000, alpha=0.05):
    """
    Simulates power for the Wilcoxon signed-rank test using actual distributions.

    Args:
      n: sample size (number of pairs).
      shift: The mean shift added to group 2.
      num_sims: Number of simulations to perform.
      alpha: Significance level.

    Returns:
        float: estimated power (proportion of simulations that correctly reject the null).
    """

    reject_count = 0
    for _ in range(num_sims):
       # Generate paired samples from two normal distributions with a defined shift.
        group1 = norm.rvs(loc=0, scale=1, size=n) # sample from N(0,1)
        group2 = norm.rvs(loc=shift, scale=1, size=n) # sample from N(shift, 1)

        #Run the wilcoxon signed rank test
        stat, p_val = wilcoxon(group1, group2, alternative='less')

        if p_val < alpha:
            reject_count += 1
    return reject_count / num_sims

# Example usage:
sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80]
powers = []

for n in sample_sizes:
    power_est = simulate_wilcoxon_power_distributional(n, shift=0.5, num_sims=500)
    powers.append(power_est)
    print(f"Sample size: {n}, Power: {power_est}")

plt.plot(sample_sizes, powers)
plt.xlabel('Sample size')
plt.ylabel('Power')
plt.title('Power curve for wilcoxon test at shift = 0.5')
plt.show()
```

Here, we simulate paired data from two normal distributions, one with a mean of zero and another with a shifted mean. This simulates an effect and can then be used to generate a power curve. Remember to change the `alternative` parameter to 'less' in our function because we simulate a shift to the *second* group.

A third approach could be to use a simplified method based on approximating the distribution of the Wilcoxon statistic under the alternative. This approximation is best explained in Lehmann's “Nonparametrics: Statistical Methods Based on Ranks”. In this approach, we first estimate the median of the differences and then use it to approximate the distribution of ranks, allowing us to compute the power. The following example is illustrative, but for very accurate calculations, one needs to dive deeper into the distribution theory:

```python
import numpy as np
from scipy.stats import norm, wilcoxon

def approx_wilcoxon_power(n, median_diff, alpha=0.05, two_sided=True):
    """
    Approximate power for the Wilcoxon signed-rank test using median difference.

    Args:
      n: sample size (number of pairs)
      median_diff: estimated median difference under alternative
      alpha: Significance level
      two_sided: Whether the test is two sided.
    Returns:
      approximate power
    """

    # Standard deviation of the Wilcoxon statistic under null
    sigma = np.sqrt(n*(n+1)*(2*n+1)/6)
    
    # Median of the Wilcoxon statistic under the alternative hypothesis
    mean_diff = n * (n + 1) * (median_diff > 0) /2

    # Calculate z critical value
    z_crit = norm.ppf(1 - alpha/2) if two_sided else norm.ppf(1 - alpha)

    # Calculate power as the probability of observing the difference at critical value
    z = (mean_diff - n*(n+1)/4 ) / sigma
    power = 1 - norm.cdf(z_crit-z)

    return power

# Example Usage
sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80]
powers = []

for n in sample_sizes:
    power_est = approx_wilcoxon_power(n, median_diff=0.4, two_sided=True)
    powers.append(power_est)
    print(f"Sample size: {n}, Approx. Power: {power_est}")
```
Here, we have a closed form expression to approximate the power based on median difference, without direct simulation. The function is a useful approximation for quick power calculations. Again, be aware of the assumptions of this function when used.

These examples should give you a solid base to begin. Key resources for understanding the underpinnings include “Nonparametric Statistical Inference” by Gibbons and Chakraborti, which provides comprehensive coverage of non-parametric methods. Lehmann’s “Nonparametrics” is another extremely authoritative resource. They detail the underlying theory that can inform the parameters you need for your simulations or approximation.

In summary, power analysis for the Wilcoxon signed-rank test is best handled through simulations, guided by a sound understanding of its underlying rank-based methodology. The effect size, especially, requires careful consideration, and you should always aim to ground it with prior knowledge about your data and the nature of the intervention you're studying. Remember, a properly powered study is the foundation of reliable results.
