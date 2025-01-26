---
title: "What is a confidence interval?"
date: "2025-01-26"
id: "what-is-a-confidence-interval"
---

Confidence intervals are foundational to statistical inference, providing a range of plausible values for an unknown population parameter, rather than just a single point estimate. This concept is crucial when working with sample data, as the sample will likely not perfectly reflect the entire population. As someone who has worked extensively on A/B testing analysis, I’ve seen firsthand the detrimental effects of misinterpreting these intervals. Ignoring their inherent probabilistic nature leads to flawed conclusions and potentially misguided business decisions.

A confidence interval quantifies the uncertainty associated with estimating a population parameter, such as the mean or proportion, from a sample statistic. It does not indicate the probability that the true population parameter lies within the calculated range. Instead, it reflects the success rate of a procedure, namely that if one were to repeat the sampling process many times, a certain percentage of the calculated intervals (determined by the confidence level) would contain the true parameter. A 95% confidence interval, therefore, means that if we were to take 100 random samples and calculate a 95% confidence interval for each, we would expect roughly 95 of those intervals to contain the true population parameter.

The calculation of a confidence interval relies on the sampling distribution of the statistic in question and a chosen confidence level. The confidence level is typically expressed as a percentage, such as 90%, 95%, or 99%, with 95% being the most common. Higher confidence levels produce wider intervals, reflecting the greater certainty we demand, but often at the cost of precision. The calculation commonly involves the sample statistic (e.g., sample mean), a critical value from a probability distribution (e.g., the z-distribution for large sample sizes, or t-distribution for small sample sizes), and the standard error of the statistic.

Consider these three examples using Python and common libraries:

**Example 1: Confidence Interval for a Population Mean (Large Sample)**

```python
import numpy as np
import scipy.stats as st

def calculate_confidence_interval_mean(data, confidence=0.95):
    """Calculates the confidence interval for the population mean, assuming a large sample size."""
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1) # Use sample std with ddof=1 for unbiased estimate
    sample_size = len(data)
    standard_error = sample_std / np.sqrt(sample_size)

    # Calculate critical value using z-distribution for large sample
    critical_value = st.norm.ppf((1 + confidence) / 2)

    margin_of_error = critical_value * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound

# Sample data of user session lengths (in seconds)
session_lengths = np.array([120, 150, 180, 200, 160, 140, 170, 220, 210, 190, 130, 165, 185, 195, 205, 155, 175, 195, 215, 145])
lower, upper = calculate_confidence_interval_mean(session_lengths, confidence=0.95)
print(f"95% Confidence Interval for Mean Session Length: ({lower:.2f}, {upper:.2f})")
```

This code defines a function to compute the confidence interval for a population mean. We calculate the sample mean and standard deviation, then the standard error. The critical value is derived from the standard normal distribution since our sample size is relatively large. The margin of error is calculated as the product of the critical value and standard error, leading to the lower and upper bounds of the interval. The output indicates that we are 95% confident that the true mean session length falls between these bounds.

**Example 2: Confidence Interval for a Population Mean (Small Sample)**

```python
import numpy as np
import scipy.stats as st

def calculate_confidence_interval_mean_small_sample(data, confidence=0.95):
    """Calculates the confidence interval for the population mean, using a t-distribution for small sample size."""
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    standard_error = sample_std / np.sqrt(sample_size)
    degrees_freedom = sample_size - 1

    # Calculate critical value using t-distribution
    critical_value = st.t.ppf((1 + confidence) / 2, degrees_freedom)

    margin_of_error = critical_value * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound

# Sample data of conversion rates from a small A/B test group
conversion_rates = np.array([0.02, 0.03, 0.025, 0.028, 0.032, 0.023, 0.029, 0.027])
lower, upper = calculate_confidence_interval_mean_small_sample(conversion_rates, confidence=0.95)
print(f"95% Confidence Interval for Mean Conversion Rate: ({lower:.4f}, {upper:.4f})")
```
This example adapts the previous one for small sample sizes. The key difference is using the t-distribution instead of the normal distribution to obtain the critical value. The t-distribution accounts for the additional uncertainty associated with small samples. The degrees of freedom are crucial for correctly determining the t-distribution shape. The result is a slightly wider confidence interval compared to what would be produced by a normal distribution for the same sample standard deviation.

**Example 3: Confidence Interval for a Population Proportion**

```python
import numpy as np
import scipy.stats as st

def calculate_confidence_interval_proportion(successes, trials, confidence=0.95):
    """Calculates the confidence interval for a population proportion."""
    sample_proportion = successes / trials
    standard_error = np.sqrt((sample_proportion * (1 - sample_proportion)) / trials)

    # Calculate critical value using z-distribution
    critical_value = st.norm.ppf((1 + confidence) / 2)

    margin_of_error = critical_value * standard_error
    lower_bound = sample_proportion - margin_of_error
    upper_bound = sample_proportion + margin_of_error

    return lower_bound, upper_bound


# Data from a survey of 500 users, with 350 expressing satisfaction.
successes = 350
trials = 500
lower, upper = calculate_confidence_interval_proportion(successes, trials, confidence=0.95)
print(f"95% Confidence Interval for User Satisfaction Proportion: ({lower:.4f}, {upper:.4f})")
```
This example calculates the confidence interval for a population proportion using a normal approximation. The inputs are the number of successful events (e.g., satisfied users) and the total number of trials (e.g., total survey respondents). We calculate the sample proportion and its standard error, then use the z-distribution to obtain the critical value. The confidence interval now gives us a plausible range for the true proportion of satisfied users in the population.

For deeper understanding, I recommend consulting resources such as *Introductory Statistics* by Neil A. Weiss or *OpenIntro Statistics* by David M. Diez, Christopher D. Barr, and Mine Çetinkaya-Rundel. Online resources like Khan Academy also offer extensive tutorials. Statistical textbooks specializing in inferential statistics can further clarify the underlying theory and practical implementation.

Here’s a comparative table to consolidate the main concepts:

| Name                          | Functionality                                                                   | Performance                               | Use Case Examples                                                                                                | Trade-offs                                                                                                   |
|-------------------------------|---------------------------------------------------------------------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Confidence Interval (Mean)    | Estimates plausible range for the true population mean.                          | Depends on sample size and data variability | Estimating average time spent on website, average income, average test score.                                  | Relies on assumptions about data distribution (normality), may widen substantially with small samples.      |
| Confidence Interval (Proportion) | Estimates plausible range for the true population proportion.                     | Depends on sample size.                   | Estimating the proportion of customers satisfied with a service, proportion of voters supporting a candidate.       | Normal approximation may not be valid for very small or extreme proportions, sensitive to sample size. |
| Bootstrap Confidence Interval  | Estimates parameter confidence interval via resampling from the original sample. | Computationally intensive.                | Cases where theoretical distributions are unclear, complex sample statistics, and non-normal data distributions. | Computationally expensive, relies on sample representativeness, stability may vary in extreme scenarios.  |

In conclusion, the optimal choice for calculating a confidence interval depends heavily on the underlying data and the specific problem. Confidence intervals for the population mean and proportion are suitable when data normality can be reasonably assumed or large sample sizes are available. However, when dealing with complex statistics, small sample sizes, or where assumptions of normality are not met, the bootstrap approach offers a flexible and powerful alternative. As a data professional, understanding the nuances of each approach and selecting the appropriate technique is pivotal for accurate interpretation and analysis.
