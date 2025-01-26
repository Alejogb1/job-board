---
title: "What is a null hypothesis?"
date: "2025-01-26"
id: "what-is-a-null-hypothesis"
---

In statistical hypothesis testing, the null hypothesis represents a default position, asserting that no effect or relationship exists within a population. It is the statement we attempt to disprove, not what we seek to prove. Based on my experience designing experiments for A/B testing frameworks, the null hypothesis forms the bedrock for assessing the statistical significance of results.

A null hypothesis, often denoted as H₀, postulates that any observed difference in sample data is merely due to random chance or sampling error, not a true underlying effect. For example, if we are testing whether a new user interface design improves click-through rates, the null hypothesis would state that there is no difference in click-through rates between the old and new designs. We are not starting from a position of assuming the new design is better; we are testing if we can reject the claim of 'no difference'. The alternative hypothesis, H₁, which we indirectly assess, proposes a meaningful difference.

The goal of hypothesis testing is to determine if there is enough evidence to reject H₀ in favor of H₁. We use statistical tests to calculate a p-value, which quantifies the probability of observing the sample data (or more extreme data), assuming the null hypothesis is true. A small p-value (typically less than 0.05) suggests that the observed data is unlikely to have occurred under H₀ and provides evidence to reject it. This rejection doesn't prove H₁ is true; it simply means we have insufficient evidence to support the null. It is crucial to understand that we cannot 'accept' the null hypothesis. Failing to reject it only indicates that the data does not provide sufficient evidence against it. The null hypothesis is the foundation upon which statistical inference is built.

Here are three examples illustrating the concept in a programming context:

**Example 1: Testing the Mean of a Dataset**

Imagine we are measuring the response times of an API endpoint and want to determine if a recent code change has altered the average response time. We formulate our null hypothesis as follows: the mean response time before the code change (μ₁) is equal to the mean response time after the code change (μ₂).

```python
import numpy as np
from scipy import stats

# Sample response times (in milliseconds)
response_times_before = np.array([120, 135, 128, 140, 115, 130, 125])
response_times_after = np.array([140, 155, 145, 160, 135, 150, 142])

# Perform an independent samples t-test
t_statistic, p_value = stats.ttest_ind(response_times_after, response_times_before)

alpha = 0.05 # Significance level

if p_value < alpha:
    print(f"P-value: {p_value:.4f}. Reject the null hypothesis. There is evidence of a change in average response time.")
else:
    print(f"P-value: {p_value:.4f}. Fail to reject the null hypothesis. There isn't enough evidence to conclude a change.")

```

In this example, we conduct a two-sample t-test to compare the means. The null hypothesis, H₀: μ₁ = μ₂, is being tested. A p-value below our significance threshold (alpha) would suggest that the observed difference in means is statistically significant, meaning it is unlikely to be due to random chance. Failing to reject does not mean there’s *no* change, just that our data can't confirm it reliably.

**Example 2: Testing Proportions with a Chi-Squared Test**

Suppose we want to test if the distribution of operating systems among users visiting our website has changed. We use a chi-squared test to check if the observed proportion of each OS differs significantly from an expected distribution. The null hypothesis would be that the observed distribution of OS usage matches our expected one; any discrepancies are due to random sampling.

```python
import numpy as np
from scipy import stats

# Observed OS counts for one week
observed_counts = np.array([200, 150, 100, 50]) # e.g., Windows, Mac, Linux, Other

# Expected proportions based on past data
expected_proportions = np.array([0.4, 0.3, 0.2, 0.1])

# Calculate expected counts based on the total observed counts
total_count = np.sum(observed_counts)
expected_counts = expected_proportions * total_count

# Perform the chi-squared test
chi2_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

alpha = 0.05

if p_value < alpha:
    print(f"P-value: {p_value:.4f}. Reject the null hypothesis. The OS distribution has changed significantly.")
else:
    print(f"P-value: {p_value:.4f}. Fail to reject the null hypothesis. No significant change in the OS distribution was detected.")
```

Here, the chi-squared test assesses how well the observed frequencies fit the expected. The null hypothesis, H₀, assumes that the observed and expected proportions are equal. A p-value lower than our significance level (alpha) indicates that the observed proportions are significantly different from the expected, leading to the rejection of H₀.

**Example 3: Testing Correlation**

We might want to determine if there is a correlation between two numerical variables within our database, for instance, the number of user logins per week and the number of support tickets created that week. Our null hypothesis states that the correlation coefficient (ρ) between these two variables is equal to zero; that is, no linear relationship exists.

```python
import numpy as np
from scipy import stats

# Sample data
logins_per_week = np.array([100, 120, 150, 130, 140, 160, 170])
support_tickets_per_week = np.array([20, 25, 35, 30, 32, 40, 45])

# Calculate the Pearson correlation coefficient and p-value
correlation_coefficient, p_value = stats.pearsonr(logins_per_week, support_tickets_per_week)

alpha = 0.05

if p_value < alpha:
    print(f"P-value: {p_value:.4f}. Reject the null hypothesis. There is a significant correlation between logins and tickets.")
    print(f"Correlation Coefficient: {correlation_coefficient:.2f}")
else:
    print(f"P-value: {p_value:.4f}. Fail to reject the null hypothesis. No significant correlation detected.")
```

In this scenario, a Pearson correlation is computed. The null hypothesis, H₀: ρ = 0, is assessed. If the p-value is below our threshold, we reject H₀, implying a correlation exists and the correlation coefficient reflects the strength and direction of the relationship.

For further understanding, I recommend exploring resources on statistical inference, hypothesis testing methods (t-tests, chi-squared tests, correlation), and the interpretation of p-values. Textbooks on introductory statistics and courses on experimental design often provide comprehensive explanations.

Here is a comparative table summarizing different common null hypotheses:

| Name                             | Functionality                                                      | Performance                                 | Use Case Examples                                            | Trade-offs                                                                                   |
| -------------------------------- | ------------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Mean Comparison (t-test)**   | Tests if means of two groups are equal.                              | Generally fast, computational cost is low. | A/B testing, comparing experiment vs control groups           |  Assumes data is normally distributed. Sensitive to outliers when dealing with small samples. |
| **Proportion Comparison (Chi-Squared)** | Tests if observed proportions match expected.             |  Generally fast, computational cost is low.        |  Analyzing categorical data such as user behavior across groups, A/B testing on proportions  | Requires adequate sample size. Sensitive to categories with small expected counts.                 |
| **Correlation Analysis (Pearson)**    |  Tests if a linear relationship exists between two variables. |  Fast and efficient, easily scalable.       | Analyzing the relationships between application metrics, user engagement.  |  Only detects linear correlations. Can be influenced by outliers.  |

In conclusion, the appropriate choice of null hypothesis and corresponding statistical test depends heavily on the nature of the data and the question being asked. For comparing averages, t-tests are ideal when assumptions of normality are met. When dealing with categorical data, chi-squared tests are suitable for assessing proportions. When quantifying the linear relationship between two numeric variables, a correlation analysis is best. Selecting the wrong test could result in misleading conclusions and should be avoided. It is always important to consult a statistical expert when in doubt, as this can impact how data is interpreted.
