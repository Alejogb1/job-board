---
title: "warning message glm.fit fitted probabilities occurred r?"
date: "2024-12-13"
id: "warning-message-glmfit-fitted-probabilities-occurred-r"
---

Alright so you got that glm fit fitted probabilities warning in R right Been there done that got the t-shirt And by t-shirt I mean I spent a solid week debugging a logistic regression model once so yeah familiar territory

This warning pops up when the `glm` function during fitting using the iterative weighted least squares algorithm hits some edge cases basically it means the predicted probabilities are either 0 or 1 or extremely close to them And yeah that causes some numerical instability and the fitting process can act a bit wonky

I've seen this thing rear its ugly head in a few scenarios usually its one of three things you've got

First up **Complete Separation** This is the most common one Lets say your dataset has a binary outcome and one of your predictors perfectly predicts this outcome like all rows where this predictor is 1 have outcome 1 and all rows where the predictor is 0 have outcome 0 The model thinks hey perfect prediction lets go to infinity and beyond The probabilities at that point get squeezed to the extreme ends and that throws the algorithm off that's the numerical instability I was talking about

Second one **Small Sample Sizes** Especially when you've got a good number of predictors relative to the data points This can lead to overfitting and that overfitting will cause the same problem as complete separation Basically its a model learning the noise and that will not give any valuable information

And then we have **Multicollinearity** when you have predictors that are correlated with each other It inflates the variance of the estimated coefficients and the coefficients will flip-flop all over the place and eventually you have the predicted probabilities trying to reach those endpoints

Okay so what do you do about it Well here are the tried and tested methods I have been through and are more effective that you think Trust me I have seen some things with this warning and this things work always if not you might have a bigger problem

Here are some code snippets to give you ideas and for you to play with

```R
# Example 1 Dealing with Complete Separation or small samples adding a ridge penalty
#   this example adds a penalty for large coefficients

library(glmnet)

#Assume data is already loaded in a dataframe called data
#Replace x with your predictors y with your target variable

# Define our variables
x <- as.matrix(data[, -which(names(data) == "target_variable")])
y <- data$target_variable

# Define grid of lambda values to be tested
lambda_grid <- 10^seq(from = -5, to = 1, length.out = 100)

# Fit the ridge regression
ridge_fit <- glmnet(x, y, family = "binomial", alpha = 0, lambda = lambda_grid)

# Using cross validation to choose the optimal lambda value
cv_ridge <- cv.glmnet(x, y, family="binomial", alpha = 0, lambda = lambda_grid)
optimal_lambda <- cv_ridge$lambda.min
best_ridge <- glmnet(x, y, family = "binomial", alpha = 0, lambda = optimal_lambda)

#Extract the coefficients of the model
coefficients_ridge <- coef(best_ridge)

#Predict probabilities with the best_ridge model
probabilities_ridge <- predict(best_ridge, newx = x, type = "response")

#Now analyze your data and see the results

```
The first snippet shows how to implement ridge regression using the glmnet package in R Ridge regression adds a penalty to large coefficients preventing them from exploding and therefore reducing the chances of extreme probability predictions This is my go-to when I suspect complete separation or small sample sizes are the issue

Okay lets go on

```R
# Example 2 Addressing Multicollinearity
#   This example uses feature selection or dimensionality reduction

# First we will use Variance Inflation Factors (VIF) to detect multicollinearity
library(car)

# Assuming data is already loaded as df replace with your dataframe
model <- glm(target_variable ~ ., data = df, family = "binomial")

# Calculate VIFs for all predictors
vifs <- vif(model)

# Print the VIFs to identify highly correlated variables
print(vifs)

# Select only variables with VIF lower than 5
variables_to_keep <- names(vifs[vifs < 5])

# Create a new data frame with only the selected variables
data_new <- df[, c(variables_to_keep, "target_variable")]

#Fit the glm with the new dataframe
model_new <- glm(target_variable ~ ., data = data_new, family = "binomial")
#Predict and analyze your new data
probabilities_new <- predict(model_new, type = "response")


# If there's still multicollinearity consider using PCA (principal component analysis)
# This example is for PCA

#Scale the data first
data_scaled <- scale(data_new[, -which(names(data_new) == "target_variable")])

#Run the PCA function
pca <- prcomp(data_scaled)
# Get the new components
pca_components <- pca$x

#Choose the first n components, depending on how much variance you want to explain
n_components <- 5
pca_data <- as.data.frame(pca_components[, 1:n_components])

#Bind the target variable to the new df with the principal components
pca_data$target_variable <- data_new$target_variable

#Run a glm model again
model_pca <- glm(target_variable ~ ., data = pca_data, family = "binomial")

#Predict and analyze your new data
probabilities_pca <- predict(model_pca, type = "response")


```
This next snippet is all about tackling multicollinearity We calculate Variance Inflation Factors VIF and eliminate highly correlated predictors if the problem persist we consider PCA to reduce the dimensionality and avoid the mess

Now this is the final trick up my sleeve
```R
#Example 3 : A simpler but yet effective solution : Adding noise to the extreme values
# A simple trick to shift those extreme values a little bit
# This will probably not be necessary in most cases after applying examples 1 or 2 but might
# be necessary in some specific cases and edge cases
probabilities <- predict(model, type = "response")

# add some small noise to the predicted probabilities
noise <- runif(length(probabilities), min = -1e-7, max = 1e-7)
probabilities_noise <- probabilities + noise

# Check and correct extreme probabilities if they are outside 0-1 range
probabilities_noise[probabilities_noise > 1] <- 1
probabilities_noise[probabilities_noise < 0] <- 0


#Now rerun your model with the new probabilities
#you can replace y by the probabilities_noise values
#and run again your logistic regression
#I am not redoing it here as the concept is shown already


```
This last snippet here is kinda the last resort When nothing else works or I am in a rush I add a tiny bit of random noise to the probabilities This shifts them away from those exact 0 or 1 values it is kinda like saying "hey model don't be so certain" Also the model might be acting weirdly cause of that noise so be aware It is a bit crude but it can do the trick in some specific cases It has saved me a couple times in the past and that is why I share it with you

Now this might sound funny but it is what happened once I spent 3 days doing feature engineering in a complex dataset only to discover that the problem was a misspelled variable name I did not even laugh I just stared at my screen for 5 min

Anyway after seeing that and trying the solutions I am sure one of the options will fit your case the best

As for further reading instead of just throwing you links out there I recommend you check out "The Elements of Statistical Learning" by Hastie Tibshirani and Friedman it is a classic and goes deep into the theory behind GLMs and regularization Another book is "Applied Predictive Modeling" by Max Kuhn this one has all you need about practical solutions for problems you can face in the industry in data science models

I've been working with GLMs in R for a while now and honestly this warning is just a common hiccup that most people go through so do not worry too much These techniques are not rocket science you just need to know when and how to apply them so yeah keep it up and do not get frustrated it will get better

Good luck!
