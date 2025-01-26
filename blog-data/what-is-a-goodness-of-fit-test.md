---
title: "What is a goodness of fit test?"
date: "2025-01-26"
id: "what-is-a-goodness-of-fit-test"
---

A goodness of fit test evaluates how well a sample distribution aligns with a hypothesized theoretical distribution. I've encountered these tests frequently in my data analysis work, particularly when validating simulations or attempting to fit real-world datasets to statistical models. The core idea revolves around quantifying the discrepancy between observed and expected frequencies. A statistically significant result from a goodness-of-fit test suggests that the observed data do not adequately follow the assumed distribution. This does not prove that the null hypothesis is false, but provides sufficient evidence to reject it. Conversely, a non-significant result does not prove the null is true, only that there is not enough evidence to reject it.

Specifically, goodness-of-fit tests assess whether the sample data could have reasonably come from a population with a specified distribution. The null hypothesis typically posits that the sample data *do* follow the assumed distribution, and the alternative hypothesis states that they *do not*. These tests are pivotal for model validation, ensuring that our underlying statistical assumptions are sound. The choice of a specific goodness-of-fit test depends on several factors including data type (continuous or categorical), the hypothesized distribution, and sample size.

Let's delve into a few common goodness-of-fit tests using Python, demonstrating their implementation with specific examples.

**Example 1: Chi-Squared Goodness-of-Fit Test**

This test is appropriate for categorical data. I often use it to verify if observed proportions of outcomes match theoretical probabilities. For instance, suppose we roll a six-sided die 60 times and record the number of times each face appears. The null hypothesis would be that the die is fair, meaning each face should appear approximately 10 times.

```python
import numpy as np
from scipy.stats import chisquare

# Observed frequencies from die rolls
observed_frequencies = np.array([8, 12, 11, 9, 10, 10])

# Expected frequencies (assuming a fair die)
expected_frequencies = np.array([10, 10, 10, 10, 10, 10])

# Perform the Chi-Squared test
chi2_statistic, p_value = chisquare(observed_frequencies, f_exp=expected_frequencies)

print(f"Chi-Squared Statistic: {chi2_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

# Interpret the results (using a significance level of 0.05)
if p_value < 0.05:
    print("Reject the null hypothesis: The die is likely biased.")
else:
    print("Fail to reject the null hypothesis: The die appears fair.")
```

In this code snippet, `scipy.stats.chisquare` efficiently calculates the chi-squared statistic and the associated p-value. A small p-value suggests that the observed frequencies deviate considerably from the expected frequencies, indicating that the null hypothesis should be rejected.

**Example 2: Kolmogorov-Smirnov (K-S) Test**

The K-S test is a powerful non-parametric test often used for continuous data. In one project, I was tasked with comparing the distribution of simulated time-to-failure data to a known Weibull distribution. It is particularly useful when you need to assess if data follows a particular continuous distribution.

```python
import numpy as np
from scipy.stats import kstest, weibull_min

# Generate simulated time-to-failure data (example)
simulated_data = np.random.weibull(2, 100) * 15

# Parameters for the theoretical Weibull distribution (shape and scale)
shape = 2
scale = 15

# Perform the K-S test against the Weibull distribution
ks_statistic, p_value = kstest(simulated_data, lambda x: weibull_min.cdf(x, shape, scale=scale))

print(f"K-S Statistic: {ks_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

# Interpret the results
if p_value < 0.05:
    print("Reject the null hypothesis: Data doesn't follow the Weibull distribution.")
else:
    print("Fail to reject the null hypothesis: Data might follow the Weibull distribution.")
```

Here, the `kstest` function from `scipy.stats` compares the cumulative distribution function (CDF) of the simulated data to that of the specified Weibull distribution. This test provides insights into how well the data conforms to a continuous, potentially non-normal, distribution.

**Example 3: Anderson-Darling Test**

The Anderson-Darling test is another option for continuous data, similar to K-S, but tends to be more sensitive to differences in the tails of the distribution. I have found it useful for assessing fit to normal distributions, particularly when outliers are suspected.

```python
import numpy as np
from scipy.stats import anderson

# Sample data (example)
data = np.random.normal(loc=5, scale=2, size=100)

# Perform the Anderson-Darling test against the normal distribution
anderson_result = anderson(data, dist='norm')

print(f"Anderson-Darling Statistic: {anderson_result.statistic:.3f}")
print(f"Critical Values: {anderson_result.critical_values}")
print(f"Significance Levels: {anderson_result.significance_level}")

# Compare the statistic to the critical values at various levels
for i in range(len(anderson_result.critical_values)):
    if anderson_result.statistic > anderson_result.critical_values[i]:
        print(f"Reject the null at level {anderson_result.significance_level[i]} %")
    else:
         print(f"Fail to reject the null at level {anderson_result.significance_level[i]} %")

```

The Anderson-Darling test uses predefined critical values corresponding to different significance levels. Comparing the calculated statistic to these critical values allows you to determine if the null hypothesis (that the data come from a normal distribution) should be rejected.

For deeper understanding, I recommend consulting these resources:
1.  "Statistical Inference" by Casella and Berger - a comprehensive text for foundational statistical theory.
2.  "All of Statistics" by Larry Wasserman - provides a practical overview of modern statistical methods.
3.  The `scipy.stats` documentation, which offers detailed explanations and usage examples for each test (the included code examples only use the base functionality.)

Here is a table summarizing the properties of these three goodness-of-fit tests:

| Name                    | Functionality                                                                  | Performance                                                                                                    | Use Case Examples                                                                                                               | Trade-offs                                                                                             |
| :---------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| Chi-Squared             | Compares observed vs expected frequencies for categorical data.                  | Generally fast, but can be less powerful for small expected frequencies in some categories.                        | Testing fairness of a die, comparing proportions in different groups, validating if survey data matches expected demographics.     | Can be inaccurate with very low expected frequencies, sensitive to the choice of bins for continuous data if discretized.   |
| Kolmogorov-Smirnov (K-S)  | Compares the empirical CDF of the data to the CDF of the hypothesized distribution. | Efficient and versatile for continuous data; sensitive to distributional differences.                         | Checking whether data is normally distributed, comparing two samples, verifying distribution assumptions in simulation studies.    | More sensitive to deviations near the center of the distribution, can struggle with very heavy tailed distributions. |
| Anderson-Darling       | Similar to K-S, but gives greater weight to the tails of the distribution.        | Typically slightly more computationally intensive, but often more powerful for tail-related deviations.          | Assessing normality, especially when outliers might be present, verifying distributions of residuals from a linear model.        | Relies more on pre-specified distribution assumptions, can over emphasize tail deviations leading to false positives.    |

In conclusion, choosing the right goodness-of-fit test hinges on the nature of your data and the specific hypothesis you are testing. The chi-squared test works well for categorical data when testing the fairness or bias of an experiment; for continuous data, the K-S or Anderson-Darling test are better suited. The K-S test is versatile for many comparisons of distributions. Anderson-Darling is often preferred to K-S when tail deviations are of particular concern, or when assessing whether the data is close to normally distributed. The choice of test should be guided by the understanding of its strengths and weaknesses in the given context, as opposed to the raw p-values. No single test is universally the best, and careful analysis should always be undertaken.
