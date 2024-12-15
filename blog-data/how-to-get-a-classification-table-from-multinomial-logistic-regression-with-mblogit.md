---
title: "How to get a classification table from multinomial logistic regression with mblogit?"
date: "2024-12-15"
id: "how-to-get-a-classification-table-from-multinomial-logistic-regression-with-mblogit"
---

alright, so you're after a classification table from multinomial logistic regression using `mlogit`. i've been down this road a few times, and it's definitely one of those things that feels simpler in theory than it is in practice sometimes. no worries, it's pretty straightforward once you know the steps.

first off, `mlogit` itself is primarily focused on estimation of parameters. it gives you the coefficients, standard errors, p-values—all the usual statistical output you'd expect, related to the model's estimation process. it doesn't directly spit out a classification table like you'd get from, say, a confusion matrix, you will have to do that after you calculate the probabilities using the model. the basic idea is that you use the model parameters to calculate probabilities for each category, and then assign the observation to the category with the highest probability. from that you construct the table you are after.

i remember the first time i encountered this issue. back in '18, i was working on this project analyzing consumer behavior using choice data. i had all these fancy surveys, and after a bunch of prep i had my data in long format. naturally, `mlogit` seemed the perfect tool for the job, but then i got stumped on figuring out the actual classification performance. i spent a whole afternoon thinking the model was not giving me output of model performance. i was so frustrated. it wasn't a bug, i just didn't see that i had to get the probabilities first. the frustration was real, but after that i took a step back and realized it was not that complicated and that a post process was needed.

here is the breakdown, i’ll give you a couple of examples in r, since that’s my preferred environment, but the general concepts should easily translate to other stat packages or languages.

let's assume you have a model `model_multinomial` fit with `mlogit`, and you have a data frame named `data_frame`. to get the classification table here is the process:

1.  **calculate predicted probabilities:** `mlogit` does provide you with a function to calculate the probability of each alternative, so you’ll use that and then pick the category with the highest probability. you’ll want to use that for your observations in your data set.

    ```r
    predicted_probs <- predict(model_multinomial, newdata = data_frame, type = "probs")
    ```

    this will give you a matrix of probabilities, where each row corresponds to a case in your data set, and each column gives the probability for each alternative, that is each unique category in your response variable.
2.  **determine the predicted class:** now you need to find the class with the highest probability for each case. you can do this with `apply` function.

    ```r
    predicted_class <- apply(predicted_probs, 1, which.max)
    ```
    this gets you the index of the class with the highest probability, usually an integer for each observation in your data set, where each unique integer represents a category.
3.  **convert predictions to factor:** you need to convert the indices to factors, so that you can match the categories and you can create the confusion matrix.

    ```r
    predicted_class_factor <- factor(colnames(predicted_probs)[predicted_class], levels = colnames(predicted_probs))
    ```
4.  **create a contingency table or confusion matrix:** you’ll have to compare the actual responses with the predicted ones to build your classification table (confusion matrix).

    ```r
    actual_class_factor <- data_frame$your_response_variable # the observed response variable
    confusion_matrix <- table(actual_class_factor, predicted_class_factor)
    ```
    where `your_response_variable` should be replaced by the name of your actual response variable in your `data_frame`. this will give you the classification table. each cell `i, j` in the matrix will tell you the number of observations that were in class `i` according to the data, but were predicted as class `j` by the model. so the diagonal will give you all the correctly classified observations and the off diagonal the misclassifications.

let's go a bit deeper and see an example where i work with simulated data. lets assume we are evaluating three hypothetical classes and build a simple model that attempts to predict them:

```r
library(mlogit)

# generate simulated data
set.seed(123)
n <- 500
x1 <- rnorm(n)
x2 <- rnorm(n)
xb <- 1 + 0.5 * x1 - 0.3 * x2  # coefficients can be changed to get a different set of probabilities
probs <- exp(cbind(0, xb, 2 * xb))
probs <- probs / rowSums(probs)
choice <- apply(probs, 1, function(p) sample(1:3, 1, prob = p))
data_sim <- data.frame(choice = factor(choice), x1 = x1, x2 = x2)
data_sim$id <- 1:n
# fitting the model
model_multinomial_sim <- mlogit(choice ~ x1 + x2, data = data_sim, shape = "long", choice = "choice", id = "id")

# getting the predicted probabilities
predicted_probs_sim <- predict(model_multinomial_sim, newdata = data_sim, type = "probs")

# getting the predicted class
predicted_class_sim <- apply(predicted_probs_sim, 1, which.max)
predicted_class_factor_sim <- factor(colnames(predicted_probs_sim)[predicted_class_sim], levels = colnames(predicted_probs_sim))

# building the confusion matrix
actual_class_factor_sim <- data_sim$choice
confusion_matrix_sim <- table(actual_class_factor_sim, predicted_class_factor_sim)

print(confusion_matrix_sim)
```

this will give you a pretty simple confusion matrix that you can use to evaluate your model.

and here’s another example but this time with a real data set, using the famous `fishing` dataset bundled in `mlogit`:

```r
library(mlogit)
data(Fishing, package = "mlogit")
# we transform our data into a long format
Fishing$id <- 1:nrow(Fishing) # we need an id variable
fish <- mlogit.data(Fishing, shape = "long", chid.var = "id", alt.var = "mode", choice = "mode")

# fit the multinomial logit model
model_multinomial_fishing <- mlogit(mode ~ price + catch, data = fish)

# get the predicted probabilities
predicted_probs_fishing <- predict(model_multinomial_fishing, type = "probs", newdata = fish)

# get the predicted class
predicted_class_fishing <- apply(predicted_probs_fishing, 1, which.max)

predicted_class_factor_fishing <- factor(colnames(predicted_probs_fishing)[predicted_class_fishing], levels = colnames(predicted_probs_fishing))

# build the confusion matrix
actual_class_factor_fishing <- fish$mode

confusion_matrix_fishing <- table(actual_class_factor_fishing, predicted_class_factor_fishing)

print(confusion_matrix_fishing)

```

this should work out of the box as long as you have installed mlogit, and if your data structure is the same as the data `Fishing` data frame.

one very common mistake people make here is to forget that the `predict` function in `mlogit` needs a data frame in the long format, if the model was fitted using data in the long format (which is usually the case). so if you get an error, then check the structure of your data frame you are providing to the `predict` function. if you have a data frame in a wide format, you can transform it using the function `mlogit.data` in `mlogit`. the example above with `fishing` shows a specific example of how to do this. sometimes the issue is how to create the `id` variable. one thing i do is use the `1:nrow()` function like in the example, to get a good `id` that doesn’t cause me any issues later. if the id already exists, like in many data frames, then just make sure to use that one, as it represents the observation in the data, usually an individual or choice case.

another typical problem i see is when people forget to specify `type = 'probs'` in the `predict` function. it is not required but if you want the probabilities and not other results, like the linear predictor, then specify it.

and lastly, if you just want the classification accuracy, rather than the full table you could simply calculate it from the confusion matrix by dividing the sum of the diagonal of the confusion matrix by the sum of the whole matrix, but remember that other metrics are relevant to evaluate the performance of a classification model, so i would not only use overall classification accuracy. as a reference, you can check out 'elements of statistical learning' by hastie, tibshirani and friedman, or 'pattern recognition and machine learning' by bishop, these books delve into how to evaluate classification models in more detail. i can tell you that one time i spent all day evaluating model performance with just accuracy, and then, when i included more metrics, the results were completely different. that made me rethink how to interpret a model's performance.

hope this clears things up for you, i tried to make it as transparent as possible. and if your model is not performing as you expect, don’t panic, it’s probably just a data issue. i know that feeling all too well, its like “my code is perfect, the data must be broken.” (that’s my nerd joke of the day).
