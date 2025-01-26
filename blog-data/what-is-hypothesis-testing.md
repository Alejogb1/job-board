---
title: "What is hypothesis testing?"
date: "2025-01-26"
id: "what-is-hypothesis-testing"
---

Hypothesis testing, a cornerstone of statistical inference, provides a structured method for evaluating claims about populations based on sample data. I've encountered numerous situations in my analytical career where correctly applying these tests was crucial to deriving meaningful insights, avoiding spurious conclusions. Essentially, it involves comparing observed data against a null hypothesis, a default assumption about the population.

The process begins with formulating both a null hypothesis (H₀) and an alternative hypothesis (H₁ or Hₐ). The null hypothesis typically represents the status quo or no effect. For instance, H₀ might state that there's no difference in average sales between two marketing campaigns. The alternative hypothesis, conversely, claims a deviation from the null, such as there *is* a difference in average sales.

Following hypothesis formulation, a suitable statistical test is chosen based on the nature of the data (e.g., continuous, categorical), the number of groups being compared (e.g., one sample, two samples), and assumptions about the data distribution (e.g., normal, non-normal). A test statistic is then calculated from the sample data. This statistic quantifies the difference between the observed data and what would be expected if the null hypothesis were true.

The calculated test statistic is subsequently compared against a critical value or used to calculate a p-value. The critical value is determined by the chosen significance level (alpha, often set to 0.05), which represents the probability of rejecting the null hypothesis when it is actually true (Type I error). The p-value, alternatively, represents the probability of observing the sample data (or more extreme data) if the null hypothesis were true. A small p-value (typically less than alpha) provides evidence against the null hypothesis.

If the test statistic falls within the rejection region (determined by critical values) or the p-value is less than alpha, the null hypothesis is rejected in favor of the alternative hypothesis. Conversely, if the test statistic falls outside the rejection region or the p-value is greater than alpha, we fail to reject the null hypothesis. Importantly, failing to reject H₀ does not mean that H₀ is true; it simply means that the data does not provide sufficient evidence to reject it.

**Code Example 1: One-Sample t-test**

I once needed to assess whether the average processing time of a new software release was significantly different from the established benchmark of 5 seconds. I used a one-sample t-test for this:

```python
import numpy as np
from scipy import stats

# Sample processing times (in seconds)
sample_times = np.array([4.8, 5.2, 5.1, 4.9, 5.3, 4.7, 5.0, 5.2, 5.1, 4.9])
population_mean = 5.0
alpha = 0.05

t_statistic, p_value = stats.ttest_1samp(sample_times, population_mean)

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The new release has a significantly different average processing time.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in average processing time.")
```

This code utilizes `scipy.stats.ttest_1samp` to conduct the one-sample t-test. I inputted the sample processing times and the population mean (benchmark) of 5 seconds. The output displays the t-statistic and the p-value. Based on comparing the p-value with the chosen alpha (0.05), I determined whether to reject the null hypothesis (no difference in average processing time) in favor of the alternative hypothesis (there is a difference).

**Code Example 2: Two-Sample t-test (Independent Samples)**

Another project involved comparing the effectiveness of two different user interface designs on task completion times. The two samples were independent, requiring a two-sample t-test assuming unequal variances:

```python
import numpy as np
from scipy import stats

# Task completion times for interface A
group_a = np.array([25, 28, 30, 32, 29, 31, 27, 33, 29, 26])
# Task completion times for interface B
group_b = np.array([22, 24, 26, 23, 27, 25, 21, 28, 24, 22])
alpha = 0.05

t_statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The two interfaces have significantly different average task completion times.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in average task completion times.")
```
In this example, `scipy.stats.ttest_ind` performs the two-sample t-test with `equal_var=False`, specifying unequal variance. The p-value again informs our decision to either reject or fail to reject the null hypothesis of no difference in average task completion times between the two interfaces.

**Code Example 3: Chi-squared Test of Independence**

I also frequently dealt with categorical data. One instance involved investigating whether there's a relationship between customer segment (e.g., new vs. returning) and product preference. I employed a Chi-squared test of independence:

```python
import numpy as np
from scipy import stats

# Contingency table: Rows are customer segments, columns are preferences
observed_values = np.array([[40, 60],  # New customers prefer A, B
                           [70, 30]]) # Returning customers prefer A, B
alpha = 0.05

chi2_statistic, p_value, _, _ = stats.chi2_contingency(observed_values)

print(f"Chi-squared statistic: {chi2_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: There is a significant association between customer segment and product preference.")
else:
    print("Fail to reject the null hypothesis: There is no significant association between customer segment and product preference.")
```

Here, `scipy.stats.chi2_contingency` performs the Chi-squared test on the contingency table representing observed frequencies. The output includes the Chi-squared statistic and the p-value. We then evaluate whether the observed relationship is statistically significant based on whether we reject the null hypothesis of independence.

**Resource Recommendations**

For a more comprehensive understanding of hypothesis testing, I recommend delving into resources such as: "Introduction to the Practice of Statistics" by David S. Moore, “Statistical Inference” by George Casella and Roger L. Berger, and materials available from universities like MIT OpenCourseware. Exploring online statistical learning platforms is beneficial as well. These resources cover theoretical foundations, assumptions underlying different tests, and diverse application scenarios with a thoroughness difficult to capture in a single response.

**Comparative Table**

| Name                     | Functionality                                                      | Performance                                               | Use Case Examples                                                       | Trade-offs                                                                                                                            |
| :------------------------ | :----------------------------------------------------------------- | :--------------------------------------------------------- | :---------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| One-Sample t-test         | Compares sample mean to a known population mean.                   | Generally efficient with small to moderate sample sizes.  | Testing if average height of a population is significantly different from a historical average, average response time of a system.       | Assumes data is approximately normally distributed. Sensitive to outliers. Requires a known population mean for comparison.     |
| Two-Sample t-test        | Compares means of two independent groups.                          | Efficient, but power decreases with unequal variances.     | Comparing the effectiveness of two teaching methods or A/B testing for user interface designs. | Assumes data is approximately normally distributed within each group. Sensitive to outliers and violations of homogeneity of variances. |
| Paired t-test            | Compares means of two related groups (e.g., before/after measurements).  | Efficient with matched pairs, but data should be truly paired. | Evaluating the impact of a drug by measuring blood pressure in the same subjects before and after treatment, testing differences between pre and post scores on a test. | Requires paired data, meaning measurements from one group correspond to particular measurements from another group.      |
| Chi-squared Test of Independence | Examines association between two categorical variables. | Works best with larger sample sizes. Results may be misleading with low expected frequencies in cells. | Investigating if there is a relationship between gender and preferred brand, a common analysis in market research. | Assumes data are categorical and observations are independent. Limited to detecting the presence of an association, not the strength or nature. |
| ANOVA  (Analysis of Variance) | Compares means of three or more independent groups.          | Efficient for multiple group comparisons, however, relies on the assumptions of normality.     | Comparing the effects of different fertilizers on crop yields, comparing user satisfaction across different platform releases.                                        | Assumes normality of residuals and homogeneity of variances across groups. Provides a global test for differences between means, but does not reveal which means differ from each other (requiring post-hoc tests).                                            |

**Conclusion**

Choosing the right hypothesis test is crucial for drawing valid conclusions. For comparing a single sample mean to a known value, or two group means, t-tests are appropriate if normality assumptions are met. ANOVA should be considered if comparing more than two group means. For evaluating associations between categorical variables, the Chi-squared test is the appropriate tool. In cases of paired measurements, the paired t-test is essential. Understanding these nuances and the inherent trade-offs is paramount for any statistician aiming to produce reliable results.
