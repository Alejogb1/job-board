---
title: "Why is lrtest() failing to perform post-hoc tests on multinomial models fitted with vglm()?"
date: "2025-01-30"
id: "why-is-lrtest-failing-to-perform-post-hoc-tests"
---
The primary reason `lrtest()` fails to perform post-hoc tests on multinomial models fitted with `vglm()` from the `VGAM` package lies in the inherent design of `lrtest()`, which is predicated on the likelihood ratio test and the direct comparison of nested model structures constructed using standard `glm()`-compatible frameworks. `vglm()`, while fitting generalized linear models, does so with a different internal representation and optimization scheme, making direct likelihood ratio comparisons incompatible with `lrtest()`. In my experience, having spent considerable time building and analyzing multinomial logistic regression models, I’ve frequently encountered this issue and had to rely on alternative approaches for post-hoc comparisons.

Specifically, `lrtest()` from the `lmtest` package expects two models passed as arguments to be nested within each other, meaning that one model is a simplified version of the other. It computes the difference in log-likelihood between the two models, along with the difference in degrees of freedom. These quantities form the basis of the chi-square test that assesses if the additional parameters in the larger model provide a significant improvement in fit compared to the smaller model. This process works seamlessly with models created by `glm()` where a consistent log-likelihood calculation and model representation are maintained.

However, `vglm()` employs a more generalized approach, potentially involving user-defined link functions and parameters not directly comparable to a standard `glm` structure. It utilizes a different internal likelihood computation routine, optimizing the vector generalized linear model using iteratively reweighted least squares. While `vglm()` models contain log-likelihood values and degrees of freedom, they are often derived using different methodologies and parameterization from that of ‘glm’, and hence cannot be directly used as if created by `glm` within `lrtest()`. Thus, feeding `vglm()` models into `lrtest()` leads to incorrect comparisons, and ultimately, produces unreliable or nonsensical results. The underlying logic of the test will not properly understand the structure of models created with ‘vglm’. The test depends on a specific structure which is only provided by specific models.

The core problem isn't that the likelihood isn't computable with models made with `vglm()`; it is that `lrtest` expects log-likelihoods computed consistently across models that conform to `glm`'s structure. As a practical matter, consider the following scenario. A researcher has three treatment groups and wants to test if the proportions of the outcome categories vary between the treatment groups. They correctly use `vglm()` to model this multinomial outcome but then incorrectly assume that post-hoc comparisons can be achieved using `lrtest()`.

Here is the first code example illustrating this:

```R
library(VGAM)
library(lmtest)

# Simulate data
set.seed(123)
n <- 300
treatment <- factor(rep(1:3, each = n/3))
outcome <- sample(1:4, n, replace = TRUE, 
                prob = c(0.2, 0.3, 0.4, 0.1))
dat <- data.frame(outcome, treatment)

# Fit the multinomial model using vglm
model_full <- vglm(outcome ~ treatment, 
                 family = multinomial(), data = dat)

# Attempt to create a nested model
model_reduced <- vglm(outcome ~ 1,
                 family=multinomial(), data=dat)

# Attempt to use lrtest to test overall significance 
# This will fail to provide proper inference
tryCatch(
  {lrtest(model_full, model_reduced)},
  error = function(e){
      message(paste("ERROR:", e))
  }
)
```

In this example, I create a dataset with three treatment groups and four possible outcome categories, simulating a typical use-case for a multinomial model.  I then fit a full model with `vglm()` that incorporates the treatment effect, and a reduced model that has no predictors. I attempt to compare them with `lrtest()`, which demonstrates that an error is returned because of the incompatible nature of the models for use in `lrtest()`. Instead of simply returning a miscalculated result it throws an error. This is better because the error indicates that the process is not compatible and prevents the researcher from relying on bad inference.

When using `vglm()`, we must rely on alternative methods for post-hoc analysis. These methods often center around manual comparisons of model fit or the use of specific packages that have explicitly considered how to extract relevant parameter information from the `vglm` object to construct tests. For instance, when the test of interest is whether a specific treatment factor has a significant impact on the outcome, we might compare nested models using `anova()` which is a method provided specifically to work with objects of class `vglm`. The function `anova()` works by internally taking into account the differences in the modeling framework.

Here is the second code example:

```R
# Using anova instead of lrtest
anova(model_reduced, model_full)
```

In this code, instead of attempting to use the incorrect `lrtest()`, I use `anova()` to achieve similar results.  `anova` automatically uses the differences in degrees of freedom and log-likelihood to create a likelihood ratio test for models of type `vglm`.  This is the proper method for conducting tests when models are produced using `vglm`.

Another approach for post-hoc testing of multinomial models built with `vglm()` when testing specific parameters is to calculate the confidence intervals for model parameters and examine if they include zero. This involves extracting the parameter estimates and their standard errors from the `vglm` object and constructing Wald confidence intervals based on asymptotic normality. While this does not directly give us the same likelihood ratio test result as `lrtest()`, it provides insight into which treatment contrasts are significant.

Here is the third code example illustrating that approach:

```R
# Confidence intervals based on asymptotic normality
coefs <- coef(model_full)
vcov_matrix <- vcov(model_full)
se <- sqrt(diag(vcov_matrix))

ci_lower <- coefs - 1.96 * se
ci_upper <- coefs + 1.96 * se
results <- data.frame(coefficient = names(coefs),
                     estimate = coefs,
                     lower_ci = ci_lower,
                     upper_ci = ci_upper)
print(results)
```

In the final example, I retrieve the estimated parameters, their variance-covariance matrix, and their standard errors. I then calculate a confidence interval for each parameter, based on the standard normal distribution approximation. This code provides the estimates for individual coefficients, including information regarding the significance of each effect by examining whether confidence intervals contain zero.

In conclusion, `lrtest()`'s incompatibility with `vglm()` models arises from differing internal calculations of likelihood values and degrees of freedom. This makes using `lrtest` directly with `vglm` a recipe for errors. For testing the significance of models made with `vglm` one must either resort to specific functions designed to work with `vglm`, like `anova()`, or alternative approaches like constructing confidence intervals around parameters of interest.

For additional resources, I recommend reviewing the vignettes provided within the `VGAM` package documentation, which offer specific examples on model fitting, diagnostics, and interpretation. Additionally, exploring statistical texts covering generalized linear models and likelihood ratio tests will improve the understanding of the underlying theoretical considerations. Finally, I have found that the online documentation for the `lmtest` package is useful for understanding the requirements for functions such as `lrtest`.
