---
title: "What are type I and type II errors?"
date: "2025-01-26"
id: "what-are-type-i-and-type-ii-errors"
---

Type I and Type II errors are fundamental concepts in statistical hypothesis testing, directly relating to the potential misinterpretations when drawing conclusions from sample data about an entire population. In my experience, a clear understanding of these errors is paramount in ensuring the validity and reliability of any data-driven decision-making process.

Essentially, a hypothesis test aims to determine whether there's enough evidence to reject a null hypothesis (H0), which typically assumes no effect or no difference. Failing to reject H0 doesn't mean it's true; it simply indicates insufficient evidence to reject it. Type I and Type II errors arise when this decision-making process results in incorrect conclusions.

A **Type I error**, often called a *false positive*, occurs when we reject the null hypothesis when it is, in fact, true. In simpler terms, we conclude that there *is* an effect or difference when there actually isn't. The probability of committing a Type I error is denoted by the Greek letter alpha (α), and is typically set to 0.05 (or 5%). This means, prior to conducting the test, we accept a 5% chance that we might falsely reject a true null hypothesis. Imagine a clinical trial for a new drug; a Type I error would conclude the drug is effective when it has no real effect.

A **Type II error**, also called a *false negative*, occurs when we fail to reject the null hypothesis when it is, in fact, false. In this instance, we conclude that there isn't an effect or difference when there actually is one. The probability of committing a Type II error is denoted by the Greek letter beta (β). The complement of beta (1-β) is known as *power*, which is the probability of correctly rejecting a false null hypothesis. For example, in the same clinical trial, a Type II error would conclude the drug is ineffective when it actually has a real benefit.

It is crucial to acknowledge that you cannot simultaneously minimize both Type I and Type II errors; there exists a trade-off between them. Decreasing the probability of one type of error tends to increase the probability of the other. This is where an informed choice of statistical methods and significance levels is essential. The choice of α typically is the result of assessing the consequence of a false positive. For example, in medical scenarios we usually see very strict α (e.g. 0.01), while for marketing A/B tests a larger value may be acceptable (e.g. 0.05). The probability of type II errors (β) on the other hand is not explicitly set. Instead, the statistical method chosen will define it. Usually, one will conduct a 'power analysis' before starting the analysis to ensure β is sufficiently small.

Here are three code examples using Python to illustrate how these concepts apply in practice, focusing on simulations since real world data analysis rarely provides us with knowledge about underlying truths:

```python
import numpy as np
from scipy import stats

# Example 1: Simulating a Type I error (False Positive)
np.random.seed(42)
true_mean = 5  # The actual mean of the population. We'll assume H0 = true_mean
sample_size = 30
alpha_level = 0.05 # The level at which we will reject the null hypothesis

# Sample from a population with the true mean
sample_data = np.random.normal(true_mean, 2, sample_size) # 2 is the standard deviation

# Perform a one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, true_mean + 1) # We test if the mean is 6

if p_value < alpha_level:
    print("Reject the null hypothesis: Type I error (false positive)")
else:
    print("Fail to reject the null hypothesis: Correct decision")

#In this example, we assume that the true mean is 5, but the test uses 6 as null hypothesis.
#Because the sample is randomly drawn from a population, we will sometimes obtain a sample
#that is statistically significantly different from 6 and conclude that H0 is wrong when
#it is actually true (since the population mean is 5).
```
This example simulates a scenario where the null hypothesis (the population mean is a specific value) is true, yet due to sampling variability, the test results lead to a rejection of the null, thus representing a Type I error. This is more likely when the sample size is small or the alpha level is higher.
```python
import numpy as np
from scipy import stats

# Example 2: Simulating a Type II error (False Negative)
np.random.seed(42)
true_mean_alt = 6  # the actual mean of the population. We'll assume the null is 5
sample_size = 30
alpha_level = 0.05

# Sample from a population with a mean slightly different from the null
sample_data = np.random.normal(true_mean_alt, 2, sample_size) # 2 is the standard deviation
# Perform a one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, 5) # null hypothesis is 5

if p_value < alpha_level:
    print("Reject the null hypothesis: Correct decision")
else:
    print("Fail to reject the null hypothesis: Type II error (false negative)")

#In this example, we assume that the population mean is 6, but the test uses 5 as null hypothesis.
#Because the sample is randomly drawn from a population, we will sometimes obtain a sample
#that is not statistically significantly different from 5 and conclude that H0 is correct when
#it is actually wrong (since the population mean is 6).
#In other words, sometimes the difference between the two means will not be big enough
#for the t-test to be able to reject the null hypothesis. This is more likely when sample
#size is small, or the alternative mean is close to the null hypothesis, or if the
#standard deviation of the population is large.
```

This example simulates a scenario where the null hypothesis is false, yet the test fails to reject it, which demonstrates a Type II error. This is influenced by factors like effect size (the difference between the hypothesized and true population parameter), sample size, and the variability of the data. A more powerful test (higher 1-β) will lead to more correct rejections.
```python
import numpy as np
from scipy import stats
import pandas as pd

# Example 3: Examining the impact of sample size on the probability of each error.
def simulate_tests(sample_size, true_mean, h0_mean, num_tests = 1000):
    type_i = 0
    type_ii = 0
    for i in range(num_tests):
      sample_data = np.random.normal(true_mean, 2, sample_size)
      t_statistic, p_value = stats.ttest_1samp(sample_data, h0_mean)

      if h0_mean == true_mean: # Null is true, test for type I
          if p_value < 0.05:
              type_i = type_i + 1
      else: #Null is wrong, test for type II
        if p_value >= 0.05:
          type_ii = type_ii + 1
    
    return (type_i/num_tests, type_ii/num_tests)

type_i_small, type_ii_small = simulate_tests(30, 6, 6) #H0 is true
type_i_large, type_ii_large = simulate_tests(300, 6, 6)
type_i_false_small, type_ii_false_small = simulate_tests(30, 7, 6)
type_i_false_large, type_ii_false_large = simulate_tests(300, 7, 6)

results = pd.DataFrame({'sample_size':[30, 300, 30, 300], 'true_mean':[6, 6, 7, 7], 'h0_mean':[6, 6, 6, 6], 'type_i_error': [type_i_small, type_i_large, type_i_false_small, type_i_false_large], 'type_ii_error':[type_ii_small, type_ii_large, type_ii_false_small, type_ii_false_large]})
print(results)

# In this example, we simulate hundreds of t-tests with differing sample sizes. By doing so, we can empirically
#verify that a larger sample size will reduce the probability of type II error, but has no effect on
#type I error.

```

This final example shows a systematic investigation into how sample size affects type I and type II errors. Specifically, by conducting multiple simulations we can verify that increasing sample size leads to smaller type II errors without affecting the rate of type I errors.

For further understanding, I recommend referring to these resources:

*   **Introductory Statistics Textbooks**: Standard textbooks on statistics and probability typically cover hypothesis testing and type I and type II errors in detail.
*   **Online Statistical Resources:** Websites such as Stat Trek or Khan Academy offer tutorials on hypothesis testing that address the topic very well.
*   **Advanced Textbooks on Experimental Design**: Books focusing on experimental design or research methodology provide valuable insight on the practical implications of error types in research.

Here's a comparative table summarizing the main aspects of type I and type II errors:

| Name          | Functionality                         | Performance                             | Use Case Examples                                                                        | Trade-offs                                                                                   |
| :------------ | :------------------------------------ | :-------------------------------------- | :--------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **Type I**    | Incorrect rejection of a true H0     | Probability = α (typically 0.05)        | Drug trial shows efficacy when there is none, detecting a non-existing cyber attack     | Increases with higher α; decreased with more conservative tests (i.e. a smaller α)           |
| **Type II**   | Failure to reject a false H0         | Probability = β (power = 1-β)         | Drug trial misses a true effect, failing to detect a real cyber attack                  | Decreases with higher α (at the expense of Type I error), increases with smaller sample size |

In conclusion, the optimal choice between minimizing Type I or Type II error depends heavily on the specific context and the consequences of each error. In high-stakes scenarios, such as medical trials, where a false positive can have severe consequences (Type I error), we often set a smaller α (e.g., 0.01) to reduce the chance of this error, even if that increases the likelihood of a Type II error. Conversely, in situations where missing a real effect might be very harmful, we could opt for a more powerful test (higher 1-β, lower β), although this may come at the cost of an increased risk of a Type I error. Ultimately, understanding the nature of both error types is fundamental in interpreting statistical results responsibly and making sound, data-informed decisions. The examples provided illustrate how the size of each error is related to the experimental design as well as the statistical methods employed.
