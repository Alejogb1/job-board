---
title: "What is a rejection region?"
date: "2025-01-26"
id: "what-is-a-rejection-region"
---

The rejection region, fundamentally, is a predefined interval or set of values within a probability distribution. I’ve encountered this concept repeatedly in my work building statistical models for financial forecasting and A/B testing. It’s a core component of hypothesis testing, representing the range of sample statistic values that, if observed, would lead to the rejection of the null hypothesis. Understanding it hinges on grasping that we're not directly proving the alternative hypothesis; rather, we're determining if the observed data is sufficiently improbable under the assumption that the null hypothesis is true.

Specifically, the rejection region is determined based on the chosen significance level, often denoted as α (alpha). This value represents the probability of rejecting the null hypothesis when it is, in fact, true – a Type I error. Common significance levels are 0.05 (5%) or 0.01 (1%), indicating a willingness to accept a 5% or 1% chance of incorrectly rejecting the null. Once α is established, its corresponding critical value(s) demarcate the boundaries of the rejection region within the relevant sampling distribution (e.g., Z-distribution, t-distribution, Chi-squared distribution). If the calculated test statistic falls inside the rejection region, we conclude the data is sufficiently unlikely under the null hypothesis, prompting us to reject it. Conversely, if the test statistic falls outside the region, we fail to reject the null, not as proof of its truth, but rather that there is insufficient evidence to disprove it. The shape and location of the rejection region depend on whether the test is one-tailed (directional) or two-tailed (non-directional), and whether we are examining population means, proportions, variances or other statistical parameters.

Let’s consider three practical coding examples to solidify this concept.

**Example 1: One-Tailed Z-Test for a Population Mean**

```python
import numpy as np
from scipy.stats import norm

# Given Data
sample_mean = 105 # observed sample mean
population_mean_null = 100 # mean under the null hypothesis
population_std = 15 # population standard deviation
sample_size = 30 # size of the sample
alpha = 0.05 # significance level

# Calculate the Standard Error of the mean
standard_error = population_std / np.sqrt(sample_size)

# Calculate the z-test statistic
z_statistic = (sample_mean - population_mean_null) / standard_error

# Calculate the critical value for a right-tailed test
critical_value = norm.ppf(1 - alpha)

# Determine if the result falls into the rejection region
reject_null = z_statistic > critical_value

print(f"Z-Statistic: {z_statistic:.4f}")
print(f"Critical Value: {critical_value:.4f}")
print(f"Reject Null Hypothesis: {reject_null}")
```
In this snippet, we're conducting a one-tailed Z-test to see if the sample mean is significantly greater than 100. We calculate the Z-statistic and compare it to the critical value obtained from the standard normal distribution at our chosen alpha.  If the z_statistic surpasses the critical value, we fall within the rejection region and reject the null hypothesis.

**Example 2: Two-Tailed t-Test for a Population Mean**
```python
import numpy as np
from scipy.stats import t

# Given Data
sample_data = np.array([54, 48, 57, 49, 52, 51, 47, 50, 53, 55]) # Sample data
population_mean_null = 50 # Mean under the null hypothesis
alpha = 0.05 # significance level
degrees_of_freedom = len(sample_data) - 1 # degrees of freedom

# Calculate the sample mean
sample_mean = np.mean(sample_data)
# Calculate the sample standard deviation
sample_std = np.std(sample_data, ddof=1)
# Calculate the standard error of the mean
standard_error = sample_std / np.sqrt(len(sample_data))
#Calculate the t-test statistic
t_statistic = (sample_mean - population_mean_null) / standard_error

# Calculate critical values for a two-tailed test
critical_value_lower = t.ppf(alpha / 2, degrees_of_freedom)
critical_value_upper = t.ppf(1 - alpha / 2, degrees_of_freedom)

# Check if the test statistic falls into the rejection region.
reject_null = (t_statistic < critical_value_lower) or (t_statistic > critical_value_upper)

print(f"T-Statistic: {t_statistic:.4f}")
print(f"Lower Critical Value: {critical_value_lower:.4f}")
print(f"Upper Critical Value: {critical_value_upper:.4f}")
print(f"Reject Null Hypothesis: {reject_null}")
```
This code performs a two-tailed t-test using sample data. The rejection region here is comprised of two areas: one at the lower end and another at the higher end of the distribution. The test statistic is compared against both critical values, and if it falls into either of these regions, the null hypothesis is rejected.  The use of the t-distribution is crucial here due to the small sample size, which makes the assumption of known population standard deviation less valid.

**Example 3: Chi-Square Test for Independence**
```python
import numpy as np
from scipy.stats import chi2

# Observed frequencies in a contingency table
observed_frequencies = np.array([[15, 20], [25, 40]])
# Calculate expected frequencies based on row and column totals
row_totals = np.sum(observed_frequencies, axis=1)
col_totals = np.sum(observed_frequencies, axis=0)
total = np.sum(observed_frequencies)
expected_frequencies = np.outer(row_totals, col_totals) / total

# Calculate the Chi-squared test statistic
chi2_statistic = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)
degrees_of_freedom = (observed_frequencies.shape[0] - 1) * (observed_frequencies.shape[1] - 1)
alpha = 0.05
critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
reject_null = chi2_statistic > critical_value

print(f"Chi-squared Statistic: {chi2_statistic:.4f}")
print(f"Critical Value: {critical_value:.4f}")
print(f"Reject Null Hypothesis: {reject_null}")
```
In this example, we use a Chi-squared test to examine the independence between two categorical variables. The rejection region is located on the right-hand side of the Chi-squared distribution. Again, we obtain a critical value based on the significance level and degrees of freedom. If our chi-squared test statistic exceeds this value, we reject the null hypothesis of independence between our categorical variables.

For further exploration, I recommend the following resources:
*   "Introductory Statistics" by OpenStax
*   "Practical Statistics for Data Scientists" by Peter Bruce, Andrew Bruce, and Peter Gedeck
*   "Statistical Inference" by George Casella and Roger L. Berger

These provide thorough explanations of the theoretical underpinnings and practical application of hypothesis testing and rejection regions.

The following table provides a comparison of the three tests discussed above, highlighting their functionalities, performance, typical use cases, and associated trade-offs.

| Name          | Functionality                            | Performance              | Use Case Examples                             | Trade-offs                                                          |
|---------------|------------------------------------------|---------------------------|----------------------------------------------|---------------------------------------------------------------------|
| One-Tailed Z-Test | Tests if a population mean is significantly greater or less than a given value.|  Computationally Efficient | Assessing the performance of a new drug is better than baseline (known std)       |  Assumes a known population standard deviation; sensitive to violations of normality  |
| Two-Tailed t-Test | Tests if a population mean is significantly different from a given value.   |   Computationally Efficient | Comparing the average sales across two different store layouts. |  Assumes normality of data; less powerful than z-test with large sample size|
| Chi-Square Test | Tests the independence between two categorical variables.| Computationally Efficient |  Determining if there is an association between marketing campaign and customer behavior | Sensitive to small expected frequencies, assumptions of independence |

Based on my experience, when dealing with large datasets and known population parameters, the Z-test can be quite efficient but also vulnerable to violations of normality if the sample size is small. The t-test, by contrast, is robust for smaller sample sizes, and works well for assessing the mean differences between sample data and a hypothesised mean. The Chi-squared test is invaluable when dealing with categorical data, although it can be quite sensitive to small values in expected frequencies. It is also essential to note that choosing the most appropriate test and defining the rejection region correctly comes from a nuanced understanding of the specific scenario at hand. Incorrect test choice or a misunderstanding of the null and alternative hypotheses could yield misleading conclusions and impact any decision that rests on the result of the test.
