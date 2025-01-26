---
title: "What is a p-value?"
date: "2025-01-26"
id: "what-is-a-p-value"
---

The p-value, a core concept in statistical hypothesis testing, represents the probability of observing data as extreme as, or more extreme than, the data actually observed, assuming that the null hypothesis is true. In essence, it quantifies the evidence against the null hypothesis, not the probability of the null hypothesis being true itself. I've seen, over years of data analysis, numerous instances where this nuance is misinterpreted, leading to flawed conclusions. Therefore, a precise understanding is vital.

A p-value is derived from the sample data and a pre-defined statistical test based on the null hypothesis (H₀). The null hypothesis is a statement of no effect or no difference, a baseline we are attempting to disprove. For instance, in a clinical trial, H₀ might be that a new drug has no effect compared to a placebo. The alternative hypothesis (H₁) is the statement that we are trying to support with the data. After collecting data, the test statistic is calculated, this metric quantifies the deviation of our observed data from what we would expect under the null hypothesis. The p-value then represents the area under the probability distribution curve of the test statistic that lies beyond the calculated test statistic value in the direction specified by the alternative hypothesis.

Essentially, the lower the p-value, the stronger the evidence against the null hypothesis. A small p-value, typically below a pre-determined significance level (α), often set at 0.05, suggests the observed data is unlikely to have arisen from random chance if the null hypothesis were true. Therefore, we have sufficient reason to reject H₀ in favor of H₁. Critically, a high p-value does not prove H₀ true; it simply indicates there is insufficient evidence to reject it. The absence of proof is not proof of absence.

Now, let’s explore some practical code examples:

**Example 1: One-sample t-test**

This Python code uses the `scipy.stats` library to perform a one-sample t-test. Imagine we're testing if the average height of seedlings under a new growth light is different from a known population average height of 10 cm.

```python
import numpy as np
from scipy import stats

# Seedling heights under the new light (sample data)
seedling_heights = np.array([10.5, 11.2, 9.8, 10.9, 11.5, 9.5, 10.2, 10.8])
# Population average height under standard conditions
population_mean = 10

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(seedling_heights, population_mean)

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a statistically significant difference.")
else:
    print("Fail to reject the null hypothesis: No statistically significant difference.")

```

Here, `stats.ttest_1samp` takes the sample data and the population mean as arguments. It calculates the t-statistic and the corresponding p-value. The code then compares the p-value to a pre-defined alpha level to determine whether to reject the null hypothesis. The p-value produced is the probability of observing a sample mean as different or more different from 10cm by chance given H0 is true.

**Example 2: Chi-square test**

In this example, we are investigating whether there's an association between two categorical variables, say, a customer's preferred website layout (layout A or B) and their purchase behavior (made a purchase or not).

```python
import numpy as np
from scipy.stats import chi2_contingency

# Observed frequencies in a contingency table
observed_data = np.array([[45, 55], [35, 65]])  # [[purchased Layout A, purchased Layout B],[did not purchase layout A, did not purchase layout B]]

# Perform chi-square test
chi2, p_value, _, _ = chi2_contingency(observed_data)

print(f"Chi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a statistically significant association.")
else:
    print("Fail to reject the null hypothesis: No statistically significant association.")

```

The `chi2_contingency` function takes the contingency table as input and returns the Chi-square statistic and the associated p-value. A small p-value would suggest the observed association is not due to chance alone.

**Example 3: Anova**

This third example tests the mean of a continuous variable across more than two groups. In this simulated scenario, we assess the impact of different fertilizers on plant growth.

```python
import numpy as np
from scipy import stats

# Plant growth data for each fertilizer
fertilizer_a = np.array([15.2, 16.1, 14.8, 15.9, 15.5])
fertilizer_b = np.array([18.3, 17.9, 19.1, 18.8, 18.5])
fertilizer_c = np.array([12.9, 13.4, 12.5, 13.8, 13.2])

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(fertilizer_a, fertilizer_b, fertilizer_c)

print(f"F-statistic: {f_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a statistically significant difference in means.")
else:
    print("Fail to reject the null hypothesis: No statistically significant difference in means.")
```

Here, `stats.f_oneway` takes samples of the variable of interest corresponding to three groups and calculates the F statistic and associated p-value. A small p-value suggests that, given H0 is true, it's unlikely the differences in sample means would be observed by random chance, thus allowing us to reject H0.

For further study, I recommend consulting resources that explain the principles of statistical inference. Introductory statistical textbooks often offer a robust foundation in hypothesis testing and p-values. Also, texts on applied statistics using Python, such as those that discuss the `scipy` library, can provide practical guidance. There are also several freely available online courses covering these topics, provided by universities and educational platforms, such as Khan Academy, Coursera and edX.

The following table summarizes the properties of the three statistical tests illustrated earlier:

| Name                     | Functionality                                                        | Performance                                                  | Use Case Examples                                                    | Trade-offs                                                                                           |
| ------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| One-Sample t-test         | Compares the mean of a single sample to a known population mean.   | Reasonably fast for small to medium sized data, relies on t distribution| Determining if a batch of manufactured items meets a specified quality standard, assessing the efficacy of a new drug against a known standard         | Assumes data is normally distributed, sensitive to outliers; does not show effect size            |
| Chi-square test          | Determines if there is a statistical association between categorical variables | computationally simple, based on large-sample assumptions       | Analyzing customer preferences by product type, exploring the relation between disease and exposure risk factors                                | Assumes sufficient sample size, does not measure the strength of association, sensitive to small sample sizes, does not show direction of relationship          |
| One-Way ANOVA           | Compares the means of more than two independent groups.               | relatively fast computationally; reliant on F distribution      | Comparing the performance of multiple marketing campaigns, analyzing the effect of different treatments on patient outcome | Assumes data is normally distributed for each group with similar standard deviations; does not identify differences between specific groups  |

In conclusion, the selection of the most appropriate test depends entirely on the research question and the nature of the data. If you are comparing a single sample mean to a known population mean, the t-test is suitable if sample data are approximately normally distributed. For examining associations between categorical variables, the Chi-square test is a commonly employed choice. When comparing means across multiple groups, ANOVA is appropriate, provided the relevant assumptions are met. It's important to remember that statistical significance, as indicated by a low p-value, does not equal practical significance. Effect size and context should always be considered when interpreting results.
