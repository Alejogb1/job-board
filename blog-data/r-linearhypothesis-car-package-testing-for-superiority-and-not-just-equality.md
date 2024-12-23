---
title: "r linearhypothesis car package testing for superiority and not just equality?"
date: "2024-12-13"
id: "r-linearhypothesis-car-package-testing-for-superiority-and-not-just-equality"
---

so you're diving into `linearHypothesis` from the `car` package right a classic I've been down this rabbit hole more times than I care to admit Let's talk about testing for superiority with `linearHypothesis` because yeah it's not as straightforward as just checking for equality

 first off `linearHypothesis` it's a fantastic tool for testing linear hypotheses but it's built primarily to assess if a set of parameters or linear combinations of parameters are equal to some value usually zero Now that's super helpful for things like checking if a certain effect is significant which often boils down to if a coefficient is different from zero or if two coefficients are equal to each other but what about what you’re asking about that is if one coefficient is *greater* or *less* than another a one sided test not just a difference

I remember early on in my career I thought `linearHypothesis` was the swiss army knife for all statistical tests I mean it handles contrasts and multi-way hypotheses so elegantly so I just assumed it did everything including one sided tests I was working on a longitudinal study comparing two different training interventions the plan was to see if the new approach was not just *different* but better I used `linearHypothesis` formulated my hypothesis as a difference test ran the `anova` and the resulting p-values were significant great but my boss came over asking "wait a minute where's the evidence of *superiority* not just difference" oh boy that's when I realized I had a problem I thought I could just get away with `p/2` trick but it turns out that's not quite right if it's not an intercept

The default behavior of `linearHypothesis` focuses on equality which is essentially a two-sided test It calculates a test statistic based on the given linear hypothesis and then computes a p-value indicating the probability of observing such a test statistic (or a more extreme one) assuming the null hypothesis which is typically equality So it does not directly tell you if one condition is better than another just if there is evidence that they are *different*

Here’s the thing you need to switch your mindset a bit Instead of asking `linearHypothesis` to test for inequality directly we'll use it to test for equality against a specific threshold then combine that with the direction of the estimate We will use one-tailed tests here and you can use `alternative=less` or `alternative=greater` within the basic `t.test` functions but you cant get this directly with `linearHypothesis` we need to make an extra step

Let's get into the code shall we Assume you have a model called `myModel` and you want to test if the coefficient of `treatment_B` is greater than that of `treatment_A`

```R
library(car)
# Assuming myModel is already fitted you have some numerical model

# First extract the coefficients
coeffs <- coef(myModel)

# Create the contrast matrix for the difference
contrastMatrix <- c(0) # initialize for the following loop
for (name in names(coeffs)){
if (name=="treatment_B"){contrastMatrix<-c(contrastMatrix,1) }
  else if (name=="treatment_A"){contrastMatrix<-c(contrastMatrix,-1)
  }
else {contrastMatrix<-c(contrastMatrix,0)}}

contrastMatrix <- contrastMatrix[-1]
contrastMatrix <- matrix(contrastMatrix, nrow = 1)


# test if equality is rejected
hyp_result <- linearHypothesis(myModel, contrastMatrix, rhs = 0)
p_value_equality <- hyp_result$`Pr(>F)`[2] #grab the P-value

# compute the estimated difference to determine which direction of the effect is significant if any
est_difference <- sum(contrastMatrix * coeffs)

# decide if it is greater than and the significance

if (est_difference > 0){
  # if est difference is positive
  p_value_superiority <- p_value_equality/2
  # if you are below alpha you have evidence of superiority
  } else {
    # if est difference is not positive we have no evidence of superiority
  p_value_superiority <- 1
}
#Now you need to check p_value_superiority with your selected alpha (e.g. 0.05)
print(paste("P-value for the equality test:",p_value_equality))
print(paste("P-value for the one-sided superiority test:",p_value_superiority))
print(paste("Estimate of B-A is",est_difference))

```
Here what's happening The `contrastMatrix` sets up a linear hypothesis that corresponds to the difference between `treatment_B` and `treatment_A` When you call `linearHypothesis` it does a standard equality test which could show a significant difference but you would still need to check if `treatment_B` is better than `treatment_A` And the magic is to check the sign of the `est_difference` then if `est_difference` is greater than 0 the p-value is cut in half to simulate the one-sided test for superiority If `est_difference` is below 0 we can not say that `treatment_B` is superior

Now lets go to another example where instead of two treatments you would be comparing one treatment with the average of other treatments:

```R
# Assuming myModel is already fitted
coeffs <- coef(myModel)

# Create contrast matrix for (treatment A) vs the average of treatment B and C
contrastMatrix <- c(0) # initialize for the following loop
for (name in names(coeffs)){
if (name=="treatment_A"){contrastMatrix<-c(contrastMatrix,1) }
  else if (name=="treatment_B"){contrastMatrix<-c(contrastMatrix,-0.5)
  }
    else if (name=="treatment_C"){contrastMatrix<-c(contrastMatrix,-0.5)
  }
else {contrastMatrix<-c(contrastMatrix,0)}}

contrastMatrix <- contrastMatrix[-1]
contrastMatrix <- matrix(contrastMatrix, nrow = 1)


# Test the contrast with linearHypothesis to reject equality
hyp_result <- linearHypothesis(myModel, contrastMatrix, rhs = 0)
p_value_equality <- hyp_result$`Pr(>F)`[2]

#compute the estimated difference to determine which direction of the effect is significant if any
est_difference <- sum(contrastMatrix * coeffs)

# decide if it is greater than and the significance
if (est_difference > 0){
  p_value_superiority <- p_value_equality/2
  } else {
  p_value_superiority <- 1
}

print(paste("P-value for the equality test:",p_value_equality))
print(paste("P-value for the one-sided superiority test:",p_value_superiority))
print(paste("Estimate of A - (B+C)/2 is",est_difference))
#Now you need to check p_value_superiority with your selected alpha (e.g. 0.05)
```

The idea remains the same except that the contrast is modified now what we are comparing is treatment A against the average effect of treatments B and C If you had more treatments it is a matter of changing the contrast matrix again

And finally something that you might think that is obvious but that you might stumble with if you are not carefull is the reference level in your model:

```R
# Assuming myModel is already fitted

coeffs <- coef(myModel)

# Create contrast matrix for (treatment B)  vs (treatment A) which is the reference
contrastMatrix <- c(0) # initialize for the following loop
for (name in names(coeffs)){
if (name=="treatment_B"){contrastMatrix<-c(contrastMatrix,1) }
  else if (grepl("treatment",name) & name!="treatment_B"){contrastMatrix<-c(contrastMatrix,0)}
  else {contrastMatrix<-c(contrastMatrix,0)}
  }


contrastMatrix <- contrastMatrix[-1]
contrastMatrix <- matrix(contrastMatrix, nrow = 1)


# Test the contrast with linearHypothesis
hyp_result <- linearHypothesis(myModel, contrastMatrix, rhs = 0)
p_value_equality <- hyp_result$`Pr(>F)`[2]

#compute the estimated difference to determine which direction of the effect is significant if any
est_difference <- sum(contrastMatrix * coeffs)

# decide if it is greater than and the significance
if (est_difference > 0){
  p_value_superiority <- p_value_equality/2
  } else {
  p_value_superiority <- 1
}
print(paste("P-value for the equality test:",p_value_equality))
print(paste("P-value for the one-sided superiority test:",p_value_superiority))
print(paste("Estimate of B - A  is",est_difference))
#Now you need to check p_value_superiority with your selected alpha (e.g. 0.05)
```

The crucial point is that the contrast matrix of `treatment_B` against `treatment_A` would look different if `treatment_A` was the reference level or any other variable was the reference level If you use the `grep` function you can avoid these errors if you are not careful with your reference levels

In my case my mistake was thinking that `linearHypothesis` automatically knew that superiority tests are important so I forgot to check the direction of the effects and only considered the p-value which was pretty embarassing on a presentation

 so to recap `linearHypothesis` is great for equality tests but when you need one-sided tests it’s a little more manual Work out your linear contrast then look at the direction of the effect and finally use one half of the p-value from the two sided test (assuming your null is centered in your selected distribution) to assess for superiority or inferiority

Resources I'd point you towards aren't specific `car` package tutorials but are more foundational texts You'd benefit from:

1.  **"Applied Regression Analysis" by Sanford Weisberg** This is a classic for understanding regression models and hypothesis testing It's pretty dense but very thorough

2.  **"Extending the Linear Model with R Generalized Linear Mixed Effects and Nonparametric Regression Models" by Julian J. Faraway** This book is great for covering many of the nuances of applying linear models in R it does have a section on contrasts and hypothesis testing

3. **"Statistical Inference" by George Casella and Roger L. Berger** This textbook is more theoretical but would clarify the ideas behind hypothesis testing and statistical inference If you are not very fond of math I would not recommend it

These aren’t just `car` tutorials but they get at the very foundation of how these tests work and why you need to consider both sides of the problem (equality AND direction) If I have to be honest with you there are no real shortcuts You need to know what's under the hood and that includes understanding hypothesis tests

One last thing because this has happened to a friend of mine dont forget that the degrees of freedom for one-tailed tests is the same as for two tailed ones you are just splitting the two sided p-value in half that does not mean that you have more statistical power If you are not careful you might misinterpret what's going on

Anyway hope that clears things up for you feel free to ping if you have any other follow-up questions
