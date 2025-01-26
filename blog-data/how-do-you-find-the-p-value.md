---
title: "How do you find the p-value?"
date: "2025-01-26"
id: "how-do-you-find-the-p-value"
---

The core concept of a p-value revolves around the probability of observing data as extreme as, or more extreme than, the current observed data, *assuming* the null hypothesis is true. My experience working with statistical modeling has shown that understanding this assumption is crucial for proper interpretation. The p-value is not the probability that the null hypothesis is true; rather, it quantifies evidence against the null. Calculating it depends on the specific statistical test being performed.

For clarity, consider a hypothetical scenario where I’m investigating whether a new fertilizer increases crop yield. The null hypothesis (H0) states that the fertilizer has no effect on yield. The alternative hypothesis (H1) claims that the fertilizer *does* affect yield (it could increase or decrease, making this a two-tailed test). After conducting the experiment, I gather yield data from both the control group (no fertilizer) and the experimental group (fertilizer applied). The process for determining the p-value involves several key steps: 1) choosing an appropriate statistical test, 2) computing the test statistic, 3) finding the p-value.

Here are three distinct examples to illustrate how p-values are calculated for different statistical tests:

**Example 1: One-Sample t-test**

Suppose I want to know if the average height of plants in my control group is statistically different from a known population average (μ = 15 cm).  I collect height data from my control group (n=25) and calculate the sample mean (x̄) and sample standard deviation (s). Let's assume x̄ = 16.5 cm and s = 2.5 cm.

```python
import numpy as np
from scipy import stats

# Sample data (hypothetical)
sample_data = np.array([16.2, 17.1, 15.8, 16.9, 16.0, 14.8, 17.3, 16.5, 15.5, 16.7,
                       15.9, 17.5, 16.3, 15.6, 16.4, 16.8, 17.0, 15.7, 16.6, 16.1,
                       15.4, 17.2, 16.0, 16.8, 16.9])
population_mean = 15
# Calculate sample statistics
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1) # ddof=1 for sample standard deviation
sample_size = len(sample_data)

# Calculate the t-statistic
t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))

# Calculate the p-value (two-tailed test)
degrees_of_freedom = sample_size - 1
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=degrees_of_freedom))

print(f"T-Statistic: {t_statistic:.3f}")
print(f"P-Value: {p_value:.3f}")
```

This code calculates the t-statistic, which measures how many standard errors the sample mean is from the population mean, under the null hypothesis. Then, it utilizes the cumulative distribution function (CDF) of the t-distribution to determine the probability of observing a t-statistic as extreme or more extreme than the calculated one. The result is the p-value. The multiplication by 2 makes it a two-tailed test, accommodating both positive and negative deviations.

**Example 2: Chi-Squared Test**

Next, I want to examine if the distribution of plant color (red, blue, green) in my experimental group differs significantly from an expected distribution. Assume I expect a 1:2:1 ratio of red, blue, green plants. My observed counts are 20 red, 45 blue, and 35 green, totaling 100 plants.

```python
import numpy as np
from scipy import stats

# Observed counts
observed_counts = np.array([20, 45, 35])
# Expected counts based on the 1:2:1 ratio
total_count = np.sum(observed_counts)
expected_counts = np.array([total_count/4, total_count/2, total_count/4])

# Calculate the chi-squared statistic
chi_squared_statistic = np.sum((observed_counts - expected_counts)**2 / expected_counts)

# Calculate the p-value
degrees_of_freedom = len(observed_counts) - 1
p_value = 1 - stats.chi2.cdf(chi_squared_statistic, df=degrees_of_freedom)

print(f"Chi-Squared Statistic: {chi_squared_statistic:.3f}")
print(f"P-Value: {p_value:.3f}")
```

The code computes the chi-squared statistic, which measures the discrepancy between observed and expected frequencies. Then, it uses the chi-squared distribution to calculate the probability of observing a chi-squared statistic as large or larger. The degrees of freedom are equal to the number of categories minus one.

**Example 3: ANOVA (Analysis of Variance)**

Suppose I have three groups of plants, each treated with different fertilizer types, and I want to determine if there's a significant difference in yields among the groups. Each group has 15 plants.

```python
import numpy as np
from scipy import stats

# Sample data (hypothetical)
group1 = np.array([25.2, 26.1, 24.8, 25.9, 25.0, 23.8, 26.3, 25.5, 24.5, 25.7, 24.9, 26.5, 25.3, 24.6, 25.8])
group2 = np.array([28.1, 29.3, 27.7, 28.8, 28.0, 27.0, 29.1, 28.3, 27.5, 28.7, 27.9, 29.5, 28.4, 27.2, 28.9])
group3 = np.array([22.5, 23.2, 22.0, 23.1, 22.3, 21.7, 23.5, 22.8, 21.9, 23.0, 22.2, 23.7, 22.6, 21.5, 22.9])


# Perform ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)


print(f"F-Statistic: {f_statistic:.3f}")
print(f"P-Value: {p_value:.3f}")
```

This code uses the `f_oneway` function from `scipy.stats` to calculate the F-statistic which measures variance between group means relative to variance within groups.  The associated p-value is then computed based on the F-distribution.  Here, I am using the F-test in ANOVA because it's designed for comparing more than two group means.

These examples highlight the variations in determining a p-value, contingent on the chosen statistical test.

**Recommended Resources**

For a deeper understanding, I'd recommend:

*   Textbooks on introductory statistics. Books that cover hypothesis testing, statistical distributions, and different test scenarios are crucial.
*   Online courses focused on statistical inference. Platforms like Coursera, edX, and Khan Academy offer excellent resources.
*   Statistical software documentation (e.g., SciPy, R). Reading documentation offers insights into the functions and algorithms used to compute the values.

**Comparative Table of Statistical Tests**

| Name                    | Functionality                                                            | Performance                                           | Use Case Examples                                                                 | Trade-offs                                                                     |
| :---------------------- | :----------------------------------------------------------------------- | :---------------------------------------------------- | :-------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| One-Sample t-test        | Compares a sample mean to a population mean.                            | Computationally efficient for reasonable sample sizes.  | Determining if average student test scores deviate from a historical average.     | Assumes data is normally distributed or sample size is large.                  |
| Chi-Squared Test          | Tests association between categorical variables.                        | Fast computation for tables with reasonable cell sizes.   | Examining whether preferences for product A, B, or C are independent of gender.   | Sensitive to small expected values in cells.                                |
| ANOVA (Analysis of Variance) | Compares means of two or more groups.                                 | Computationally efficient, especially with balanced datasets. |  Comparing the effects of different fertilizers on crop yield.                   | Assumes equal variances across groups and data should be approximately normal.  |

**Conclusion**

Selecting the optimal approach depends on the data characteristics and research question. The one-sample t-test is best when evaluating a single sample against a known population mean. The chi-squared test is appropriate for evaluating relationships between categorical data. ANOVA is ideal for comparing means across multiple groups. If normality assumptions cannot be met for the t-test or ANOVA, non-parametric alternatives, such as the Mann-Whitney U or Kruskal-Wallis test, should be considered.  Understanding each tests’ assumptions and limitations is vital for correct interpretation and application.
