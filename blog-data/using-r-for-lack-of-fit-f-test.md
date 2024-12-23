---
title: "using r for lack of fit f test?"
date: "2024-12-13"
id: "using-r-for-lack-of-fit-f-test"
---

 so you're asking about using an F-test for lack of fit right Been there done that it's a classic check when you're trying to fit a model to some data and you wanna know if your model is actually any good or if it’s just kinda pretending

Lets just jump right in I've wrestled with this more times than I care to admit especially early in my career I remember this one project oh man that thing was a beast We were trying to model the thermal behavior of some semiconductor device and we were using this polynomial regression model because someone decided it'd be a good idea because its a function and functions are cool right Anyway the fit looked  at first glance the r-squared was decent but something just felt off you know like that gut feeling you get when your code is compiling but it just feels wrong Yeah I had that feeling so I decided to dig deeper I went ahead and started going through the usual checks and I remember thinking "I bet theres a lack of fit" and boom that suspicion turned out to be true That whole thing was painful I was up until 3am doing these calculations by hand that was before I learned how to really leverage r It was a mess but lesson learned we switched to a more appropriate model a physically based one instead of just blindly trusting an equation the lesson for me was the importance of understanding the limits of your models

So here’s the deal the F-test for lack of fit is all about seeing if the variation in your data that your model is not able to capture is just random noise or if it's something systematic something that your model is straight up missing It compares the variation of data around your regression model with the variation of your data around the mean when your variables are grouped or repeated for same x values its a fancy way of saying "are we really close to our data or are there trends our model cant explain"

The fundamental idea boils down to this you compare two sources of variation: the *lack of fit* part which is the difference between your observed values and your modeled values when each independent variable (x) is repeated several times, and the *pure error* part which is just the variation that you see between the repeated data at the same independent value the more variation is explained by the difference between groups than your model predicts the greater this test will be This is not something you get by default in most statistical software

Here is the basic procedure.

First you need to build your model and fit it to the data let’s assume for a second that we are working with simple linear regression the formula is like y = m*x + c you would do your model building there using a library like "stats" in R.

Second you need to identify repeated values if you only have 1 value for each x then this F test is not what you need you should test something else like residual plots.

Third you will need to calculate sums of squares the sums of squares due to regression which are the variations explained by your model, the sums of squares due to error which are the variations that are not explained by your model and finally we split the error sum of square into two parts the sum of squares due to lack of fit and the sum of squares due to pure error These are obtained by a few calculations as follow

*   **SSR (Sum of Squares Regression):** Variation explained by your model.
*   **SSE (Sum of Squares Error):** Variation not explained by your model this is the usual sum of the squared residuals.
*   **SSLOF (Sum of Squares Lack of Fit):** Variation not explained by the model by considering each group
*   **SSPE (Sum of Squares Pure Error):** Pure variation inside each group

Then you will compute the mean square of each component as the sums of squares divided by the degrees of freedom the mean squares values are important because they allow for comparison between the sums of squares which are biased by the number of observations involved in the calculation.

*   **MSR (Mean Square Regression):** SSR / degrees of freedom of the model
*   **MSE (Mean Square Error):** SSE / degrees of freedom of the error
*   **MSLOF (Mean Square Lack of Fit):** SSLOF / degrees of freedom lack of fit
*   **MSPE (Mean Square Pure Error):** SSPE / degrees of freedom pure error

Finally the F statistic can be computed as the ratio of the mean square of lack of fit over the mean square of pure error F = MSLOF / MSPE The value of the F statistic will tell us if the lack of fit is significant or not.

Now for the nitty gritty R code part let's look at some examples

```R
# Example 1 Basic lack of fit test
# Simulate some data
set.seed(123)
x <- rep(1:5, each = 3) # Repeated values
y <- 2 * x + rnorm(length(x), mean = 0, sd = 2) # Some linear data with some noise

# Fit a linear model
model <- lm(y ~ x)

# Lack of fit test function (you might find a built-in function in some packages but let's build our own for the example)
lack_of_fit_test <- function(model, x, y) {
    y_hat <- fitted(model) # Predicted values
    unique_x <- unique(x)  # Groups of the same x value
    
    SSLOF <- 0
    SSPE <- 0
    
    for(val in unique_x){
        group_y <- y[x == val]
        group_y_hat <- y_hat[x == val]
        group_mean <- mean(group_y)
        
        n <- length(group_y) # number of repetition
        
        SSLOF <- SSLOF + n*(group_mean - group_y_hat[1])^2
        
        SSPE <- SSPE + sum((group_y - group_mean)^2)
    }
    
    dfe <- length(y) - length(coef(model)) # Degrees of freedom for the model
    dfl <- length(unique_x) - length(coef(model))
    dfpe <- length(y) - length(unique_x)
    
    MSLOF <- SSLOF / dfl
    MSPE <- SSPE / dfpe
    
    F_stat <- MSLOF / MSPE
    p_val <- pf(F_stat, dfl, dfpe, lower.tail = FALSE)

  return(list(F_statistic = F_stat, p_value = p_val, degrees_freedom_lack_of_fit = dfl, degrees_freedom_pure_error = dfpe))
}

# Perform lack of fit test
result <- lack_of_fit_test(model, x, y)

# Print the test result
print(result)

#Interpretation: p-value should be very close to 1 which means the test is not significant
# since our data is a straight line and our fit should be good.
```
Now if the result shows a p-value less than 0.05 then there’s a good chance your model is missing something important and you should be looking for a more appropriate model but in this case it will most probably be higher since we simulated our data to be a linear function and our model can fit it pretty well

Here is a slightly more complicated example where a simple linear model is not a good idea for the simulated data

```R
# Example 2 Lack of fit with curvature

# Simulate some data with a curve
set.seed(456)
x <- rep(1:5, each = 3)
y <- 2 * x + 0.5 * x^2 + rnorm(length(x), mean = 0, sd = 3)

# Fit a linear model
model <- lm(y ~ x)

# Use the same lack_of_fit_test function from before
result <- lack_of_fit_test(model, x, y)

# Print the test results
print(result)

# Interpretation: p-value here is most probably below 0.05 which would mean
# that there is a significant lack of fit

#Now try fitting a quadratic regression this should eliminate the lack of fit
model_quadratic <- lm(y~x+I(x^2))
result_quadratic <- lack_of_fit_test(model_quadratic, x, y)

# Print the test results
print(result_quadratic)
# The p-value should be very high meaning the lack of fit is not significant since
# the model can actually explain the data well

```

Now a question I get asked a lot is "what if you don't have repeated measurements at the same x values?" well then you have a problem my friend because the F test won't really work as it needs the pure error variation that is calculated from those repeated measurements without them the error term is really all you got and your lack of fit will be impossible to test accurately.

So if the data doesnt have repeated x values the F-test for lack of fit will produce an error in R since it will result on the division by zero as the degrees of freedom of pure error will be zero as there are no variations inside the groups but lets simulate this situation to make this more clear.

```R
# Example 3 lack of fit with no repeated values

# Simulate data with no repeated x
set.seed(789)
x <- 1:15
y <- 2*x + 0.5*x^2 + rnorm(length(x), mean = 0, sd = 5)

# Fit a linear model
model <- lm(y ~ x)
# Try the lack of fit test
result <- try(lack_of_fit_test(model,x,y), silent = TRUE)
# This will return an error because there is no pure error to compare with

if (inherits(result,"try-error")){
    print("Error: Lack of fit cannot be tested because there is no repeated measurements")
} else {
    print(result)
}


# This is because you have no variation within each value of x. You need at least two values of y for each value of x to calculate the pure error term.
# When you don't have replicated data, you would rely on residual plots and other types of tests, not the F-test lack of fit
```

Now the thing about models is that they’re just fancy approximations of reality and its your job to know when they are not valid. If your data shows signs of systematic error or your lack of fit test comes back screaming the model isn't good then you know something is wrong.

For deeper dives I would recommend reading "Applied Linear Statistical Models" by Kutner et al its a classic and goes over these concepts in great detail also "The Statistical Sleuth" by Ramsey and Schafer is another good resource for applied statistical modeling and these types of checks or if you are into something more theoretical go for "Statistical Inference" by Casella and Berger which contains a lot of theoretical insight.

Oh and speaking of deeper dives did you hear about the statistician who had an existential crisis? He kept asking himself "What if I'm just a large sum of squared errors" I know corny right? But seriously understanding the error is key with statistics.
And that's basically it Use the F-test wisely and don’t just trust the p-values blindly be critical with the models you use and always check assumptions this is crucial when building statistical models it’s easy to make mistakes but being cautious and aware of the tests that are available will greatly improve your ability to analyze data.
