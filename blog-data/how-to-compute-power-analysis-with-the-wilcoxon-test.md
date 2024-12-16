---
title: "How to compute power analysis with the Wilcoxon test?"
date: "2024-12-16"
id: "how-to-compute-power-analysis-with-the-wilcoxon-test"
---

Alright, let's tackle this. Computing power analysis with the Wilcoxon test—a non-parametric alternative to the t-test—can indeed feel a bit less straightforward, and I’ve certainly seen its challenges firsthand. It's not as simple as plugging values into a pre-existing formula like you might for a t-test. Over the years, I've had to implement this several times, sometimes in rather complex scenarios, and I've developed a methodology that’s usually pretty reliable. I remember one particularly tricky project involving clinical trial data where parametric assumptions were clearly violated, leaving us with no choice but to use a Wilcoxon-based approach.

The core issue is that power analysis fundamentally involves knowing or estimating the distribution of your test statistic under both the null hypothesis (usually that there’s no effect) and a specific alternative hypothesis (that there is some effect of a certain magnitude). With parametric tests like the t-test, we often can assume a normal distribution, which makes life considerably easier. With Wilcoxon, we're working with ranks, not the original data themselves, and the distribution of these ranks is more complex and depends on the sample size and specific alternative distribution shape.

So, how do we go about it? It generally breaks down into these steps:

1.  **Define the Alternative Hypothesis:** First, you need to specify what kind of effect you’re trying to detect. This isn't simply "there’s a difference," but rather how big of a difference you expect. Unlike t-tests where you might define a mean difference, here, you need to think about the shift in the underlying distribution. For instance, you could consider a stochastic dominance scenario, where the distribution of one group is shifted to the right of the other. A common practical approach is to define the difference between means or medians under the assumption of the underlying shift.

2.  **Simulate Data:** This is where the "non-parametric" aspect really kicks in. You'll simulate data from both groups based on your alternative hypothesis. Since we're using the Wilcoxon test, the actual distributions of each group are less critical than the differences in their central tendencies (remember, we're dealing with ranks). You’ll need to create two samples where their distributions have the type of shift you previously defined.

3.  **Run the Wilcoxon Test:** Perform the Wilcoxon signed-rank test (for paired data) or Wilcoxon rank-sum test (for independent groups) on the simulated data. This gives you a single test statistic and a p-value.

4.  **Repeat Many Times:** Repeat steps 2 and 3 many times (usually thousands of times). You’ll store the resulting p-values each time.

5.  **Calculate Power:** Finally, the power is the proportion of simulations where the resulting p-value is less than your chosen significance level (alpha, typically 0.05). In simple terms, how often would you correctly reject the null hypothesis of no effect, given a specific effect size and alpha?

Now, let's look at this in some actual Python code with `scipy`. I'll start with independent samples.

```python
import numpy as np
from scipy.stats import ranksums
from tqdm import tqdm # Used here to track progress.

def wilcoxon_power_independent(sample_size, shift, num_simulations=1000, alpha=0.05):
    """
    Estimates power of the Wilcoxon rank-sum test for independent samples.

    Args:
        sample_size (int): Sample size for each group.
        shift (float): The shift in the mean of the second group relative to the first.
        num_simulations (int): The number of simulations to perform.
        alpha (float): The significance level.

    Returns:
       float: The estimated power.
    """

    p_values = []
    for _ in tqdm(range(num_simulations), desc='Simulating'):
        group1 = np.random.normal(0, 1, sample_size) # Assume normal for simplicity.
        group2 = np.random.normal(shift, 1, sample_size) # Shifted normal.
        statistic, p_value = ranksums(group1, group2)
        p_values.append(p_value)

    power = sum(np.array(p_values) < alpha) / num_simulations
    return power


# Example usage:
sample_size = 50
shift = 0.5
power_estimate = wilcoxon_power_independent(sample_size, shift)
print(f"Estimated power: {power_estimate}")

```

This `wilcoxon_power_independent` function calculates the power for two independent groups. You can modify the distributions used in `np.random.normal` to use another distribution if that's more appropriate to your use case. Crucially, the `shift` parameter directly affects the "effect size." The larger the shift, the higher the power will be. We’re approximating power by looking at how often we correctly reject the null hypothesis in our simulations. `tqdm` helps track the progress, especially when you use a large `num_simulations` value.

Next, let's look at a similar function for paired samples:

```python
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm

def wilcoxon_power_paired(sample_size, shift, num_simulations=1000, alpha=0.05):
    """
    Estimates power of the Wilcoxon signed-rank test for paired samples.

    Args:
        sample_size (int): Number of paired observations.
        shift (float): The average difference of the paired observations.
        num_simulations (int): Number of simulations to perform.
        alpha (float): The significance level.

    Returns:
        float: The estimated power.
    """
    p_values = []
    for _ in tqdm(range(num_simulations), desc='Simulating'):
        paired_group1 = np.random.normal(0, 1, sample_size)
        paired_group2 = paired_group1 + np.random.normal(shift, 0.5, sample_size)  # The shift occurs as an *added* component
        differences = paired_group2 - paired_group1
        statistic, p_value = wilcoxon(differences) # Wilcoxon signed-rank on paired data
        p_values.append(p_value)


    power = sum(np.array(p_values) < alpha) / num_simulations
    return power


# Example usage:
sample_size = 30
shift = 0.3
power_estimate = wilcoxon_power_paired(sample_size, shift)
print(f"Estimated power: {power_estimate}")
```

Here, we use the `wilcoxon` function, which implicitly performs the signed-rank version because we’re giving it the *differences* as input. It's important to realize that in the paired case, the shift is introduced as a shift to each observation from the same set which affects the distribution of differences. This mirrors the reality that paired observations are correlated. Note that we used two independent sources of randomness, with one having a larger variance than the other so that the differences do not follow a degenerate distribution.

Finally, it's quite useful to visualize the relationship between these parameters. We can visualize the estimated power with different sample sizes for the same effect size:

```python
import matplotlib.pyplot as plt
import numpy as np

sample_sizes = np.arange(20, 100, 10) # sample sizes 20 to 90 in steps of 10.
shift = 0.5 # fixed shift for illustration

powers = [wilcoxon_power_independent(s, shift) for s in sample_sizes]

plt.plot(sample_sizes, powers, marker='o')
plt.title("Power vs. Sample Size for Wilcoxon Test")
plt.xlabel("Sample Size (per group)")
plt.ylabel("Estimated Power")
plt.grid(True)
plt.show()
```

As you would expect, power increases with sample size.

Now, for resources, I'd strongly recommend diving into "Nonparametric Statistical Methods" by Hollander, Wolfe, and Chicken. It’s a dense text but the authority for all things non-parametric. Another fantastic resource is "Practical Nonparametric Statistics" by Conover, which provides both theory and application. For simulation-based approaches to power analysis, it is worth reviewing the literature around Monte Carlo techniques, and papers around the specific area of statistical power. These books should give you a solid background that transcends the simplicity of formula-based power analysis.

Remember, with non-parametric tests, you're often going to have to rely on simulation because closed-form solutions are rare. Simulation helps understand the performance of these tests when parametric assumptions are violated. It's not just an exercise of 'plugging in numbers' but a way to gain practical insights into how much power you really have to detect the effect you’re interested in.
