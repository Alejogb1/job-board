---
title: "How can cross-validation be implemented in R without using the caret package?"
date: "2024-12-23"
id: "how-can-cross-validation-be-implemented-in-r-without-using-the-caret-package"
---

Alright, let's tackle cross-validation in R without relying on `caret`. It's a fundamental technique, and knowing how to implement it from scratch is invaluable, regardless of which abstraction layers you usually work with. I've been through this process numerous times, particularly back in my early days when project constraints sometimes precluded using specific packages, or when I simply needed a deeper understanding of the underlying mechanisms. It's not rocket science, but it requires careful attention to detail.

We're essentially trying to build a process that accurately estimates the performance of a model on unseen data. The core idea of cross-validation involves partitioning your dataset into multiple subsets; a model is trained on some subsets and evaluated on the remaining subset, and we then repeat this procedure multiple times while cycling the used data partitions. This helps average out variations in model performance that might arise from the specific choice of training data. Let's dive in.

Fundamentally, we need to manage the data partitioning and model fitting process. The most common methods are k-fold cross-validation and leave-one-out cross-validation (loocv), each appropriate for different situations. In k-fold cross-validation, the dataset is partitioned into `k` equal-sized subsets or "folds." The model is trained `k` times, each time using a different fold as the test set and the remaining folds as the training set. With loocv, you essentially have as many folds as there are samples; the model is trained using all data except one, which is used as a test. This process is repeated for every sample.

Let’s start with a practical example: k-fold cross-validation, with k set to 5. We’ll use a simple linear regression model for demonstration. This approach is beneficial when you have a reasonably large dataset.

```r
k_fold_cross_validation <- function(data, k = 5, model_formula) {
    n <- nrow(data)
    folds <- sample(rep(1:k, length.out = n))
    results <- data.frame(fold = numeric(0), mse = numeric(0))

    for (i in 1:k) {
        test_data <- data[folds == i, ]
        train_data <- data[folds != i, ]

        model <- lm(model_formula, data = train_data)
        predictions <- predict(model, newdata = test_data)
        mse <- mean((test_data[, all.vars(model_formula)[1]] - predictions)^2)

        results <- rbind(results, data.frame(fold = i, mse = mse))
    }

    return(results)
}

# Example Usage:
set.seed(123)
data <- data.frame(
  y = rnorm(100, mean = 5, sd = 2),
  x1 = rnorm(100, mean = 10, sd = 3),
  x2 = rnorm(100, mean = -5, sd = 1)
)
model_formula <- y ~ x1 + x2
cv_results <- k_fold_cross_validation(data, k = 5, model_formula = model_formula)
print(cv_results)
mean(cv_results$mse)
```

This script defines a function `k_fold_cross_validation` which takes a data frame, the number of folds `k`, and a model formula as input. Within the function, it randomly assigns each data row to one of the `k` folds. For each fold, it constructs a training and test data set, fits the linear regression model, computes the mean squared error, and stores the result. The function returns a data frame with each fold’s error, and we subsequently calculate the mean of the MSE across all folds.

Now, let's look at leave-one-out cross-validation. This is particularly helpful when dealing with smaller datasets, but it can be computationally more expensive because the model must be trained as many times as there are samples.

```r
leave_one_out_cv <- function(data, model_formula) {
  n <- nrow(data)
  mse_values <- numeric(n)

  for (i in 1:n) {
    test_data <- data[i, , drop = FALSE] # Ensures we keep it as a data frame
    train_data <- data[-i, ]

    model <- lm(model_formula, data = train_data)
    prediction <- predict(model, newdata = test_data)
    mse_values[i] <- (test_data[, all.vars(model_formula)[1]] - prediction)^2
  }

  return(data.frame(index = 1:n, mse = mse_values))
}

# Example Usage
set.seed(456)
data <- data.frame(
  y = rnorm(20, mean = 3, sd = 1),
  x1 = rnorm(20, mean = 8, sd = 2)
)

model_formula <- y ~ x1
loocv_results <- leave_one_out_cv(data, model_formula)
print(loocv_results)
mean(loocv_results$mse)
```

The function `leave_one_out_cv` iterates through each row in the data set. For every row it designates the row as the test set and the remaining rows as the training set. The procedure is the same as before; the mean squared error is calculated. The function returns the mean squared error for each sample, and we average the output MSE for the final performance metric. Note the `drop = FALSE` argument to maintain the result as a data frame when indexing single rows. This avoids subtle bugs that can otherwise occur with data frame indexing.

Finally, let's address a scenario where we might need stratified k-fold cross-validation, which is used when you have unbalanced classes in your classification task. This approach ensures that each fold maintains the same class distribution of the whole dataset. We will make a minor change to the k-fold code.

```r
stratified_k_fold_cv <- function(data, k = 5, model_formula, class_col) {
  classes <- unique(data[[class_col]])
  folds_per_class <- lapply(classes, function(class_value) {
    indices <- which(data[[class_col]] == class_value)
    n_indices <- length(indices)
    fold_assignments <- sample(rep(1:k, length.out = n_indices))
    data.frame(index = indices, fold = fold_assignments)
  })

  all_folds <- do.call(rbind, folds_per_class)
  results <- data.frame(fold = numeric(0), mse = numeric(0)) # Adjust metric as necessary.
    for (i in 1:k) {
        test_indices <- all_folds$index[all_folds$fold == i]
        train_indices <- all_folds$index[all_folds$fold != i]

        test_data <- data[test_indices, ]
        train_data <- data[train_indices, ]

    model <- lm(model_formula, data = train_data)
    predictions <- predict(model, newdata = test_data)

        mse <- mean((test_data[, all.vars(model_formula)[1]] - predictions)^2) # Adjust metric as needed.
      results <- rbind(results, data.frame(fold = i, mse = mse))
    }
    return(results)
}


# Example Usage (with binary classification - although we calculate MSE for demonstration)
set.seed(789)
data <- data.frame(
  y = c(rep(0, 60), rep(1, 40)), # Imbalanced classes
  x1 = rnorm(100, mean = ifelse(c(rep(0,60), rep(1,40))==0, 5, 10), sd = 3),
  x2 = rnorm(100, mean = ifelse(c(rep(0,60), rep(1,40))==0, -2, 2), sd = 1)
)

model_formula <- y ~ x1 + x2
class_col <- "y"

cv_results <- stratified_k_fold_cv(data, k = 5, model_formula = model_formula, class_col = class_col)
print(cv_results)
mean(cv_results$mse)
```

The `stratified_k_fold_cv` function first identifies each unique class label. Then, for each class, it distributes data indexes into `k` folds. Finally, we combine these class-specific fold assignments. The rest of the code follows the previous k-fold setup. Notice that the core function is still the same but has a slightly different fold assignment process to account for stratification.

These examples demonstrate the fundamental building blocks of cross-validation in R. For further reading, consider delving into "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, a fantastic resource for a deep understanding of machine learning concepts. Also, the book "Applied Predictive Modeling" by Kuhn and Johnson provides excellent practical guidance on model validation techniques. Furthermore, the journal of Statistical Software has several articles and papers covering implementations of this process in R as well as statistical considerations. Finally, understanding concepts from linear model theory as well as basic optimization routines will add depth to the statistical performance of the validation. These resources should be invaluable to anyone seeking a more comprehensive understanding of this process. It’s not just about using a library; it’s about understanding the underlying methodology.
