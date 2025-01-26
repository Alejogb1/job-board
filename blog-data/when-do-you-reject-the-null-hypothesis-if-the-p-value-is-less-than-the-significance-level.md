---
title: "When do you reject the null hypothesis if the p-value is less than the significance level?"
date: "2025-01-26"
id: "when-do-you-reject-the-null-hypothesis-if-the-p-value-is-less-than-the-significance-level"
---

Statistical hypothesis testing hinges on the comparison of the p-value to a predefined significance level, commonly denoted as alpha (α). A p-value less than the significance level provides the justification to reject the null hypothesis, a core principle I've encountered repeatedly in A/B testing scenarios and model evaluation.

**Explanation of the Principle**

The null hypothesis (H₀) is a statement of no effect or no difference that we aim to disprove. Conversely, the alternative hypothesis (H₁) posits that an effect or difference does exist. The p-value is the probability of observing the data, or more extreme data, if the null hypothesis were true. The significance level (α) is the probability of rejecting the null hypothesis when it is actually true; this is known as a Type I error. It's a threshold set by the researcher. Common values are 0.05, 0.01, and 0.10, representing a 5%, 1%, and 10% risk of committing a Type I error, respectively.

If the p-value is less than α (p < α), we interpret this as the observed data being sufficiently unlikely under the assumption that the null hypothesis is true. Consequently, we deem the evidence strong enough to reject H₀ in favor of H₁. This doesn't "prove" H₁ but indicates there’s enough statistical evidence to support it over H₀. Conversely, if the p-value is greater than or equal to α (p ≥ α), we fail to reject the null hypothesis. This does *not* mean that the null hypothesis is true, but rather that the data do not provide sufficient evidence to reject it.

The critical interplay between p-value and alpha is essentially a risk management strategy. Alpha determines our tolerable risk of a Type I error, and the p-value assesses the probability of observed data under H₀. When the p-value falls below this risk threshold, we proceed with rejecting H₀.

**Code Examples with Commentary**

Here are three code examples demonstrating this principle using Python with `scipy.stats` for testing:

```python
# Example 1: Comparing means of two independent samples with a t-test
import numpy as np
from scipy import stats

# Sample data for two groups
group_a = np.array([56, 72, 68, 75, 61])
group_b = np.array([78, 84, 79, 88, 82])

# Performing a two-sample t-test
t_statistic, p_value = stats.ttest_ind(group_a, group_b)

# Significance level
alpha = 0.05

print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: There's a significant difference between group A and group B.")
else:
    print("Fail to reject the null hypothesis: There's no significant difference between group A and group B.")

# Commentary: This example tests for a difference in mean between two groups.
# The t-test calculates a t-statistic and corresponding p-value.
# The conditional statement then compares the p-value against alpha.
# If the p_value is sufficiently small (below 0.05), we reject the null hypothesis of no difference between group means.
```

```python
# Example 2: Chi-squared test for categorical data
import numpy as np
from scipy import stats

# Observed frequencies
observed = np.array([[20, 30], [40, 10]])

# Performing the Chi-squared test
chi2_statistic, p_value, dof, expected = stats.chi2_contingency(observed)

# Significance level
alpha = 0.01

print(f"Chi-squared statistic: {chi2_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: There is a significant association between the variables.")
else:
     print("Fail to reject the null hypothesis: There's no significant association between the variables.")
# Commentary:  Here, a Chi-squared test examines the association between two categorical variables, testing if the observed counts deviate significantly from expected values under the null hypothesis of independence.
# The decision to reject the null is based on whether the p-value exceeds our chosen risk threshold, in this case 0.01.
```

```python
# Example 3: Performing a one-sided Z-test for a population mean
import numpy as np
from scipy import stats

# Sample data, population mean under H0, and population standard deviation
sample_mean = 52.0
pop_mean = 50.0
pop_std = 4.0
sample_size = 36

# Calculate the z-statistic
z_statistic = (sample_mean - pop_mean) / (pop_std / np.sqrt(sample_size))
p_value = 1 - stats.norm.cdf(z_statistic) #one-sided test for greater than

# Significance level
alpha = 0.05

print(f"Z-statistic: {z_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The population mean is significantly greater than 50.")
else:
    print("Fail to reject the null hypothesis: The population mean is not significantly greater than 50.")

# Commentary: This demonstrates a one-sided (right-tailed) z-test.
# We test if a sample mean is significantly greater than a hypothesized population mean, utilizing a known standard deviation.
# Again, we reject the null when the p-value falls below our predefined alpha of 0.05, thus concluding a statistically significant increase in the population mean.
```

**Resource Recommendations**

For further exploration of statistical hypothesis testing and p-values, several resources are valuable. "All of Statistics" by Larry Wasserman offers a comprehensive theoretical foundation. "Practical Statistics for Data Scientists" by Peter Bruce, Andrew Bruce, and Peter Gedeck provides a more applied, hands-on approach suitable for data analysis. Finally, standard statistical textbooks from authors like Moore, McCabe, and Craig provide excellent fundamentals.

**Comparative Table**

| Name              | Functionality                                                                  | Performance                       | Use Case Examples                                                | Trade-offs                                                                                                                             |
|-------------------|-------------------------------------------------------------------------------|------------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| **T-test**        | Compares means between two groups.                                              |  Relatively fast for moderate-size datasets.        | A/B testing to check for differences in conversion rates; comparing treatment groups in experiments.   | Assumptions of normality and equal variance must be met; sensitive to outliers. Limited to comparing two groups.  |
| **Chi-squared**   | Tests the association between categorical variables or goodness of fit.         | Computationally efficient even with large samples. | Assessing relationships between survey responses; analyzing the results of genetic crosses.       | May require large sample sizes; cannot determine direction of association; sensitive to small cell counts.                                                                      |
| **Z-test**        | Tests hypotheses about a population mean with known population standard deviation. | Highly efficient with large samples.               | Testing sample averages against a fixed population mean; quality control checks.                     | Requires a large sample size; population standard deviation must be known. May be less appropriate when sample size is small.                                                                |

**Conclusion**

The choice of statistical test and, therefore, how the p-value is used for decision-making depends on the nature of the data and the research question. T-tests are well-suited for comparing means of two independent groups, while Chi-squared tests are essential for categorical data and assessing independence. Z-tests, while powerful for large samples with known standard deviations, are less often applicable in typical business analysis where the population standard deviation is often unknown. In practice, understanding the assumptions of each test and carefully selecting the appropriate method are critical. For smaller sample sizes and unknown population standard deviation, I would typically use a t-test, while for categorical variable assessments, chi-squared tests are more suitable. The principle remains consistent: a p-value below alpha justifies the rejection of the null hypothesis, signaling a statistically significant effect, but also highlighting the necessity of interpreting these results with respect to context and potential for error.
