---
title: "What is the likelihood ratio test?"
date: "2025-01-26"
id: "what-is-the-likelihood-ratio-test"
---

The likelihood ratio test (LRT) is a statistical hypothesis test that compares the likelihood of two competing statistical models, one nested within the other, given the observed data. The test quantifies how much more likely the data is under the full, more complex model compared to the simpler, constrained model. This comparison is crucial in model selection, providing a formalized method to assess whether the additional complexity of a model offers a statistically significant improvement in fit. Having used this technique extensively in building predictive models for financial markets, I've observed its practical impact on deciding between models with varying parameter complexities.

Here’s a more detailed explanation:

The fundamental principle revolves around the concept of likelihood. For a given statistical model, the likelihood represents how probable the observed data is, given the model's parameters. A higher likelihood signifies a better fit between the model and the data. The LRT leverages this concept by constructing a likelihood ratio, typically denoted as lambda (λ). This ratio is calculated by dividing the maximized likelihood of the constrained model (L₀) by the maximized likelihood of the full model (L₁):

λ = L(θ₀ | data) / L(θ₁ | data)

Where θ₀ represents the parameters of the constrained model and θ₁ represents the parameters of the full model.  The constraint here is that the parameter space for θ₀ is a subset of the parameter space for θ₁.  For instance, θ₁ may include additional parameters or less constrained parameters compared to θ₀.

Because the constrained model, by definition, cannot fit the data as well as the full model, λ will always be less than or equal to 1.  A value closer to 1 suggests the constrained model fits the data almost as well as the full model, while a value substantially less than 1 suggests the full model provides a better fit.

To make statistical inference, we usually employ the negative logarithm of this ratio (-2 * ln(λ)), also known as the test statistic, denoted by G or sometimes by D. Under the null hypothesis (that the constrained model is adequate), this test statistic, G, is asymptotically distributed as a Chi-square distribution with degrees of freedom equal to the difference in the number of free parameters between the two models (i.e., degrees of freedom = |parameters in θ₁ - parameters in θ₀|).  This distribution is used to obtain a p-value for the LRT.

The p-value reflects the probability of observing a test statistic at least as extreme as the one calculated if the null hypothesis (the constrained model) were true. If this p-value is below a predefined significance level (e.g., 0.05), we reject the null hypothesis in favor of the alternative hypothesis, indicating the full model provides a statistically significant improvement.

Now let us explore some code examples using Python, assuming the availability of relevant libraries like `statsmodels` for statistical modeling and hypothesis testing, and `numpy` for numerical computation.

**Code Example 1: Comparing Linear Regression Models**

```python
import numpy as np
import statsmodels.api as sm

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 2)  # Two predictor variables
X = sm.add_constant(X) # add a constant for the intercept
y = 2 * X[:, 1] + 3 * X[:, 2] + np.random.normal(0, 1, 100)  # Generate response variable

# Model 1 (constrained): Using only the first predictor
model1 = sm.OLS(y, X[:, [0, 1]]).fit()

# Model 2 (full): Using both predictors
model2 = sm.OLS(y, X).fit()

# Perform the likelihood ratio test
lr_test_stat = -2 * (model1.llf - model2.llf)
df = model2.df_model - model1.df_model
p_value = 1- stats.chi2.cdf(lr_test_stat, df)

print(f"LR Test Statistic: {lr_test_stat:.3f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.3f}")
```
*Commentary:* This example demonstrates comparing a linear regression model with one predictor (constrained) against a model with two predictors (full).  The `statsmodels.api` library is used for OLS fitting, and the test statistic is computed based on the difference in log-likelihood values (`llf`) between the two models.  The Chi-squared CDF is used to calculate the p-value given the test statistic and degrees of freedom which is the difference in number of parameters between the two models.

**Code Example 2: Comparing Generalized Linear Models (Poisson)**
```python
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

# Generate sample count data for poisson regression
np.random.seed(42)
X = np.random.rand(100, 2) # Two predictors
X = sm.add_constant(X)
mu = np.exp(1 + 2 * X[:, 1] + 3 * X[:, 2])  # Underlying rate for Poisson
y = np.random.poisson(mu)  # Observed count data

# Model 1 (constrained): Poisson regression with one predictor
model1 = sm.GLM(y, X[:, [0, 1]], family=sm.families.Poisson()).fit()

# Model 2 (full): Poisson regression with both predictors
model2 = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Calculate the likelihood ratio test statistic
lr_test_stat = -2 * (model1.llf - model2.llf)
df = model2.df_model - model1.df_model
p_value = 1- stats.chi2.cdf(lr_test_stat, df)


print(f"LR Test Statistic: {lr_test_stat:.3f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.3f}")
```

*Commentary:* This example extends to generalized linear models (GLMs) using a Poisson distribution, commonly used for count data. Similar to the linear regression, the code compares two Poisson models (with and without the second predictor) using `statsmodels` and performs the likelihood ratio test.

**Code Example 3: Nested Logistic Regression Models**

```python
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

# Generate sample binary data
np.random.seed(42)
X = np.random.rand(100, 2)
X = sm.add_constant(X)
probability = 1 / (1 + np.exp(- (1 + 2* X[:, 1] + 3 * X[:, 2] )))
y = np.random.binomial(1, probability)

# Model 1 (constrained): Logistic regression with one predictor
model1 = sm.Logit(y, X[:, [0, 1]]).fit()

# Model 2 (full): Logistic regression with both predictors
model2 = sm.Logit(y, X).fit()

# Calculate the likelihood ratio test statistic
lr_test_stat = -2 * (model1.llf - model2.llf)
df = model2.df_model - model1.df_model
p_value = 1- stats.chi2.cdf(lr_test_stat, df)


print(f"LR Test Statistic: {lr_test_stat:.3f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.3f}")

```

*Commentary:*  This example demonstrates the use of LRT for comparing two nested logistic regression models. It follows the same pattern as the previous examples, calculating the likelihoods and deriving the test statistic.

**Resource Recommendations**

For a deeper understanding of the theoretical underpinnings of the likelihood ratio test, I would recommend exploring textbooks and online materials focusing on:
1. **Statistical Inference:**  Look for books that provide a detailed explanation of hypothesis testing, focusing on parametric tests and asymptotic theory, often found in courses or texts covering statistical modeling.
2.  **Generalized Linear Models:** Learning GLM concepts is essential when the response variable is not normally distributed. Texts or online course notes on GLM and its extensions would be highly beneficial.
3. **Mathematical Statistics:**  Understanding of concepts like maximum likelihood estimation and asymptotic distributions will be crucial, so texts that deal with statistical theory are necessary.
4. **Programming Guides for Statistical Libraries:** Explore the documentation of libraries such as `statsmodels` in Python, as they provide concrete instructions and examples of using the likelihood ratio test function.

**Comparative Analysis**

Here's a table that compares the LRT with other common hypothesis testing methods:

| Name | Functionality | Performance | Use Case Examples | Trade-offs |
|---|---|---|---|---|
| Likelihood Ratio Test | Compares the likelihood of nested models | Asymptotically Chi-squared; efficient for nested model comparison | Model selection, comparing linear models, GLMs, mixed models | Relies on nested models; can be computationally demanding for complex models |
| Wald Test | Tests hypotheses about individual parameters in a model | Based on asymptotic normality of parameter estimates; fast to compute | Assessing significance of individual predictors in a regression | Can be less accurate with small samples; might not converge for complex models |
| Score Test (Lagrange Multiplier Test) | Tests hypotheses by evaluating the gradient of the likelihood function | Can be advantageous when only the constrained model needs to be fit | Assessing whether a variable should be included in a model | Requires derivatives of the likelihood function; can be computationally intensive for certain model types |
| F-test | Compares the variance of two nested linear models | Based on F-distribution, exact test for linear models | Compares two nested linear regression models | Only applicable to linear models with normally distributed errors, is a special case of the LRT|

**Conclusion**

The Likelihood Ratio Test provides a powerful tool for comparing nested statistical models, enabling us to make informed decisions about model complexity. While tests like the Wald test and F-test are valuable in specific scenarios, the LRT stands out for its generality across various model types (linear, GLM, etc.). However, it's crucial to remember that the LRT relies on asymptotic properties and may not perform ideally with small sample sizes.

In scenarios where the goal is to compare nested models and the computational cost isn't prohibitive, I’ve often found the LRT to be the most principled choice. This is particularly true when evaluating the statistical significance of adding complexity to models within financial forecasting or risk management. In contrast, if the primary interest is assessing specific parameter significance, the Wald test is often the quicker option. However, if model selection is the objective, and model nesting is satisfied, I've consistently found the likelihood ratio test to provide a statistically sound approach.
