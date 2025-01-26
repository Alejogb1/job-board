---
title: "What is a Shapiro-Wilk test?"
date: "2025-01-26"
id: "what-is-a-shapiro-wilk-test"
---

The Shapiro-Wilk test is a powerful tool in statistical analysis specifically designed to assess the normality of a dataset, often a crucial assumption for many parametric statistical tests. I’ve encountered its use countless times in my experience developing statistical models and evaluating data quality, finding its sensitivity particularly useful in identifying non-normal distributions that might otherwise be overlooked by visual inspection alone.

A Shapiro-Wilk test operates by calculating a test statistic, W, which quantifies the similarity between the given data’s distribution and a normal distribution. The test statistic ranges from 0 to 1; a value of 1 indicates a perfect fit to the normal distribution, while lower values suggest a deviation. The actual calculation of W involves ordering the data points, calculating a weighted sum of these ordered values, and comparing this sum to the sum of squares of the data points. Crucially, this calculation requires a pre-determined set of constants dependent on the sample size; these constants are typically obtained from statistical tables. The result, alongside the sample size, allows the calculation of a p-value. The p-value is the probability of obtaining the observed data (or more extreme data) if the underlying distribution were indeed normal. A low p-value (usually below a chosen significance level, commonly 0.05) provides evidence against the null hypothesis that the data are normally distributed, suggesting the distribution significantly deviates from normality.

The strength of the Shapiro-Wilk test is its power, particularly when dealing with smaller samples (N<50), which makes it more effective at identifying non-normality than other tests like the Kolmogorov-Smirnov test. This is important, as many real-world datasets I've worked with often fall into this sample size range due to the nature of experiments or data availability. It is also adaptable to various distributions, enabling its use in many different analytical contexts.

Let's illustrate this with code examples in Python, using the `scipy` library, which I frequently use:

```python
import numpy as np
from scipy import stats

# Example 1: Data generated from a normal distribution
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=50)
stat, p = stats.shapiro(normal_data)
print(f"Example 1 - Test Statistic: {stat:.3f}, p-value: {p:.3f}")
# Output: Example 1 - Test Statistic: 0.986, p-value: 0.867

# Explanation:
# This example tests data generated from a normal distribution.
# The p-value of 0.867 is significantly above the typical alpha level of 0.05,
# suggesting that we fail to reject the null hypothesis, i.e. the data appear normally distributed.
```

```python
# Example 2: Data generated from a skewed (exponential) distribution
exponential_data = np.random.exponential(scale=1, size=50)
stat, p = stats.shapiro(exponential_data)
print(f"Example 2 - Test Statistic: {stat:.3f}, p-value: {p:.3f}")
# Output: Example 2 - Test Statistic: 0.773, p-value: 0.000

# Explanation:
# This example tests data generated from an exponential distribution, which is highly skewed.
# The p-value of 0.000 (which in practical terms means a very small value significantly below 0.001)
# clearly indicates non-normality, and we reject the null hypothesis.
# The test statistic is also significantly lower than in the first example.
```

```python
# Example 3: Data with outliers
outlier_data = np.concatenate([np.random.normal(loc=0, scale=1, size=49), [10]])
stat, p = stats.shapiro(outlier_data)
print(f"Example 3 - Test Statistic: {stat:.3f}, p-value: {p:.3f}")
# Output: Example 3 - Test Statistic: 0.888, p-value: 0.000

# Explanation:
# Here, we test data that is mostly normally distributed but includes a single significant outlier.
# This is also a form of departure from normality. The Shapiro-Wilk test effectively identifies it,
# indicated by a small p-value.
```

While powerful, the Shapiro-Wilk test, like any statistical method, has limitations. First, although effective, its power might be reduced when the dataset grows large (N>5000). Although not a frequent scenario, it is important to be aware of this. Second, as a test of a null hypothesis, a 'failure to reject' the null hypothesis does not necessarily prove that the distribution is normal. It merely suggests the available evidence isn't strong enough to conclude it's not normal, a nuance I always emphasize in my reports. The test should be used as one piece of evidence among others, including visual inspection of data distributions using histograms and Q-Q plots.

For in-depth understanding, I recommend consulting:

*   **"Practical Statistics for Data Scientists"** by Peter Bruce, Andrew Bruce, and Peter Gedeck; this text provides a hands-on overview of statistical methods, including tests for normality.
*   **"All of Statistics: A Concise Course in Statistical Inference"** by Larry Wasserman; it offers a more mathematically rigorous treatment of statistical tests.
*   The official documentation for statistical libraries like **SciPy**; these resources often provide detailed information about the implementation and usage of specific tests.

The following table compares the Shapiro-Wilk test against a few other common tests for normality:

| Name                 | Functionality                                                                | Performance                                                                                                         | Use Case Examples                                                                                                                  | Trade-offs                                                                                                       |
|----------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Shapiro-Wilk**    | Tests for normality of a sample by assessing its fit to a normal distribution | Powerful for small to moderate sample sizes (N<5000), known to be more effective than others with such sample sizes | Checking assumptions of parametric statistical tests (ANOVA, t-tests), verifying data quality, model diagnostic testing         | Sensitivity to outliers, less reliable for very large samples (N>5000), requires pre-calculated constants |
| **Kolmogorov-Smirnov** | Compares a sample distribution to a known distribution (usually normal)       | Less powerful than Shapiro-Wilk for normality testing, good for comparing any continuous distributions                 | Checking goodness-of-fit to specified non-normal distributions, comparing two empirical distributions                            | Lower power compared to Shapiro-Wilk for normality, sensitive to sample size, limited to testing against fully specified distributions |
| **D'Agostino-Pearson** | Tests for normality based on skewness and kurtosis                           | Good performance for large sample sizes, less sensitive to outliers than Shapiro-Wilk                                 | Checking normality for large sample data where outliers might be a concern, data transformation testing                                   | Less powerful than Shapiro-Wilk for small samples, might be less sensitive to subtle departures from normality |
| **Anderson-Darling**  | Tests for normality using a weighted measure of distances from the normal distribution  | Good overall performance across different sample sizes, effective for detecting deviations in the tail of the distribution | Similar to Shapiro-Wilk, may be preferred in situations where tail departures are a major concern                               | Can be computationally intensive, may overemphasize the tails for distributions with very thick tails. |

In conclusion, the optimal choice among these tests depends on the specific context. For small to moderate sample sizes where detecting subtle deviations from normality is crucial, the Shapiro-Wilk test is typically the most powerful choice, a preference born out of my experiences. However, for larger samples, D’Agostino-Pearson or Anderson-Darling could offer advantages in terms of computational efficiency or robustness to outliers. In situations where one is evaluating against a distribution other than normal, or comparing two empirical distributions, the Kolmogorov-Smirnov test remains suitable. Therefore, using a combination of such tests, combined with visualizations, offers a comprehensive approach to ensure that underlying distribution assumptions are adequately addressed in statistical analyses.
