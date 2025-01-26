---
title: "What is an alternative hypothesis?"
date: "2025-01-26"
id: "what-is-an-alternative-hypothesis"
---

In the realm of statistical inference, the alternative hypothesis (denoted as H₁) posits a relationship or effect that directly contradicts the null hypothesis (H₀). The null hypothesis typically assumes no effect or no difference. My experience designing A/B testing platforms has frequently brought me face-to-face with both of these core concepts. Failing to properly formulate the alternative hypothesis can lead to erroneous conclusions and flawed decision-making, therefore understanding its nuances is crucial.

The alternative hypothesis essentially becomes the research hypothesis that we seek to support through statistical evidence. It’s the "something is happening" scenario that we try to find sufficient evidence for, whereas the null hypothesis is the "nothing is happening" position that we're aiming to reject. Crucially, the nature of the alternative hypothesis dictates the type of statistical test used and how results are interpreted. Depending on our research question, the alternative hypothesis may be directional (one-tailed) or non-directional (two-tailed). A directional alternative hypothesis specifies the direction of the difference or effect (e.g., the mean is *greater than* or *less than*), while a non-directional hypothesis simply states there's a difference without specifying the direction (e.g., the mean is *different from*). The choice of directional or non-directional alternative impacts the p-value calculation and consequently, the conclusion of hypothesis testing.

Let's illustrate with some practical code examples, drawing from my experience with data analysis and statistical modeling.

**Code Example 1: One-Sample T-Test (Two-Tailed)**

This example uses a one-sample t-test to determine whether a sample mean is significantly different from a known population mean. This illustrates a non-directional alternative hypothesis.

```python
import numpy as np
from scipy import stats

# Generate a sample of data (simulated click-through rates)
sample_data = np.array([0.05, 0.06, 0.07, 0.04, 0.08, 0.05, 0.06, 0.07, 0.06, 0.05])
population_mean = 0.06  # Assume this is the known average CTR

# Perform a two-tailed one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, population_mean)

# Define the null and alternative hypotheses
# H0: Sample mean is equal to the population mean.
# H1: Sample mean is not equal to the population mean. (Two-Tailed)

alpha = 0.05  # Significance level

print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to show a significant difference.")
```

Here, the alternative hypothesis (H₁) is that the sample mean is *not equal to* the population mean. The t-test output determines if the data provides sufficient evidence to reject the null hypothesis. The code directly translates the statistical concept into an actionable analysis.

**Code Example 2: One-Sample T-Test (One-Tailed - Greater Than)**

Here's a case of a directional alternative, specifically to ascertain if a sample mean is *greater than* a specified value. This example showcases a one-tailed test.

```python
import numpy as np
from scipy import stats

# Generate another set of sample data (simulated conversion rates)
sample_data = np.array([0.10, 0.12, 0.11, 0.13, 0.14, 0.11, 0.12, 0.13, 0.12, 0.11])
population_mean = 0.11  # Previous average conversion rate

# Perform a one-tailed one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, population_mean, alternative='greater')

# Define the null and alternative hypotheses
# H0: Sample mean is equal to or less than the population mean
# H1: Sample mean is greater than the population mean (One-Tailed)

alpha = 0.05  # Significance level

print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis. There is significant evidence the mean is greater.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to show the mean is greater.")

```

In this case, the alternative hypothesis (H₁) states that the sample mean is *greater than* the population mean. The `alternative='greater'` argument specifies the one-tailed direction in the `stats.ttest_1samp` function. Failing to account for this directional hypothesis can lead to missed significant results.

**Code Example 3: Chi-Squared Test**

My work with user segmentation has often relied on chi-squared tests to assess association between categorical variables. This example demonstrates how an alternative hypothesis is constructed in that scenario.

```python
import numpy as np
from scipy.stats import chi2_contingency

# Observed frequencies of user behaviour
observed_values = np.array([[30, 70], [60, 40]])

# Perform the chi-squared test
chi2, p, dof, expected = chi2_contingency(observed_values)

# Define the null and alternative hypotheses
# H0: Two variables are independent
# H1: Two variables are dependent (Associated)

alpha = 0.05 # significance level

print(f"Chi-squared Statistic: {chi2:.2f}")
print(f"P-value: {p:.3f}")


if p < alpha:
    print("Reject the null hypothesis. There is a significant association.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to show a significant association.")
```

In this example, the alternative hypothesis (H₁) asserts that there is an *association* between two categorical variables, while the null hypothesis (H₀) states they are independent. The Chi-Squared test evaluates if observed data deviates sufficiently from what's expected under the null hypothesis.

For deeper understanding and practical application of hypothesis testing, I recommend the following resources: “Introductory Statistics” by OpenStax, “Practical Statistics for Data Scientists” by Peter Bruce, Andrew Bruce, and Peter Gedeck, and “The Book of Why” by Judea Pearl. These provide a solid foundation in both the theory and application of statistical methods.

**Comparative Table of Alternative Hypothesis Types**

| Name                 | Functionality                                                                | Performance                                       | Use Case Examples                                                                     | Trade-offs                                                                                                                    |
|----------------------|------------------------------------------------------------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Two-Tailed**        | Tests for a difference without specifying direction (H₁: μ ≠ X).               | Lower statistical power compared to one-tailed tests |  Comparing two groups’ means; testing for a treatment effect without assuming its direction.| More conservative, less likely to commit a Type I error (false positive), but can fail to detect a smaller, directional effect.   |
| **One-Tailed (Greater than)**| Tests if a value is significantly greater than another (H₁: μ > X)       | Higher statistical power if the true effect is in the specified direction.| Testing if a new method has *better* performance; checking if a treatment has *increased* a specific metric.| Higher risk of a Type II error (false negative) if the effect is in the opposite direction of what was specified, or if there's no effect at all.|
| **One-Tailed (Less than)** | Tests if a value is significantly less than another (H₁: μ < X).         | Higher statistical power if the true effect is in the specified direction. | Testing if a website redesign *decreased* bounce rates; checking if a campaign had a *negative* impact on sales.| Higher risk of a Type II error (false negative) if the effect is in the opposite direction, or if there's no effect at all.|

In summary, the choice of alternative hypothesis is deeply influenced by your research questions and prior knowledge. If you have a specific directional hypothesis, then a one-tailed test may be appropriate and increase the likelihood of detecting that difference, however, only if your effect aligns with the specified direction.  If there's no strong basis for a directional hypothesis, a two-tailed test remains the prudent default. Improper alternative hypothesis specification can lead to incorrect inference and, consequently, bad decisions. Therefore, a precise definition is critical for rigorous statistical analysis.
