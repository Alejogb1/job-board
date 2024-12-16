---
title: "How to run a stepwise regression on a folder full of CSVs using lapply?"
date: "2024-12-16"
id: "how-to-run-a-stepwise-regression-on-a-folder-full-of-csvs-using-lapply"
---

Alright, let's talk stepwise regression, and specifically how to tackle that across a folder of csv files using `lapply` in R. I remember facing a similar challenge a few years back when I was tasked with analyzing a large dataset of sensor readings from a distributed network. Each sensor had its own csv file, and I needed to build a predictive model for each one, which, of course, required variable selection. Stepwise regression felt like a reasonable approach at the time, although, let's be honest, we all know its limitations now. Still, for certain preliminary investigations, it's useful.

The core idea, as we all know, is to automate a process, not do it manually for each file. The `lapply` function is fantastic for this type of task, as it allows you to apply a function over a list—in this case, our list of file paths—and collect the results. It's a clean and efficient way to handle file processing. I usually avoid for loops when it comes to operations on lists in R, because they aren’t nearly as elegant and can easily become a debugging nightmare.

Before diving into code, it's important to note some potential pitfalls. Stepwise regression, whether forward, backward, or both, is a data-driven process. That means it can be sensitive to idiosyncrasies in the data, like outliers or multicollinearity, and is not great at discovering new variables. It also often leads to models that overfit to the current data, which makes it critical to assess them out-of-sample before deploying them anywhere. Consider that I might be using it as an initial step, not the final solution.

Now, let's get into the practical application. First, we need a function to handle the regression process for a single csv.

```r
perform_stepwise_regression <- function(file_path, response_variable, predictor_variables, alpha_enter = 0.05, alpha_remove = 0.05) {
  # Load the data
  data <- read.csv(file_path)

  # Ensure the response variable and predictors exist in the data
  if(!all(c(response_variable, predictor_variables) %in% names(data))){
    warning(paste("Skipping file", file_path, ". Response or predictor variables not found."))
    return(NULL)
  }

  # Construct the formula string
  formula_string <- paste(response_variable, "~", paste(predictor_variables, collapse = "+"))
  formula <- as.formula(formula_string)


  # Attempt a full model first; if it doesn't work return null
  tryCatch({
    full_model <- lm(formula, data = data)
  }, error = function(e){
    warning(paste("Skipping file", file_path, ". Could not fit base model."))
      return(NULL)
  })

  # Check that lm worked as expected.
  if (!exists("full_model")) {
      warning(paste("Skipping file", file_path, ". Full model failed to build."))
      return(NULL)
  }



  # Perform stepwise regression using step function.
  stepwise_model <- step(full_model,
                          direction = "both",
                          scope = list(lower = as.formula(paste(response_variable, "~ 1")),
                                    upper = formula),
                           trace=0,
                            k=qchisq(1-alpha_enter, 1, lower.tail = TRUE),
                           steps=1000)


  # Extract model details
  summary_data <- summary(stepwise_model)
  coefficients <- coef(stepwise_model)
  r_squared <- summary_data$r.squared
  adj_r_squared <- summary_data$adj.r.squared

  # Return a list with file info and relevant model outputs
  return(list(file_path = file_path,
              model = stepwise_model,
              coefficients = coefficients,
              r_squared = r_squared,
              adj_r_squared = adj_r_squared
            )
         )
}
```

This function does a few crucial things. It reads the csv file into a dataframe, constructs the formula, performs the stepwise regression using R's `step()` function, and finally, returns the results—the fitted model itself, as well as key details such as model coefficients and r-squared values. Notably, I use `tryCatch` to handle cases where the data for a given file is problematic and the base regression can't be fitted, because you don’t want one bad file to crash the whole process.

Now, for the second piece, we need to use the `lapply` function to apply the above function to all files in a folder.

```r
run_stepwise_on_folder <- function(folder_path, response_variable, predictor_variables, alpha_enter=0.05, alpha_remove=0.05) {
  # Get the list of csv files in the folder
  file_list <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)

  # Apply the regression function to each file path using lapply
  results <- lapply(file_list, function(file) {
    perform_stepwise_regression(file, response_variable, predictor_variables, alpha_enter=alpha_enter, alpha_remove=alpha_remove)
  })

  # Return the results
  return(results)
}
```

This `run_stepwise_on_folder` function will take a folder path, response, and predictor variables, and loop through each CSV file using `lapply` passing all inputs to our helper function, and returning a named list with information about every regression run.

Let's wrap it up with a concrete example of how you might execute this. Let’s assume you have a directory named "sensor_data," containing numerous CSV files, each with a response variable named "target_variable" and predictors such as "predictor1", "predictor2", and "predictor3".

```r
# Set parameters
folder_path <- "sensor_data"
response_variable <- "target_variable"
predictor_variables <- c("predictor1", "predictor2", "predictor3")

# Run regression on the folder
results <- run_stepwise_on_folder(folder_path, response_variable, predictor_variables)


# Filter out Null results; these indicate files where errors occurred
results <- results[!sapply(results, is.null)]


# View summaries for the first three models for demonstration (if they exist)
if(length(results)>0) {
    for(i in 1:min(length(results),3)){
      cat(paste("Summary for file", results[[i]]$file_path, ":\n"))
      print(summary(results[[i]]$model))
      cat("\n")
    }
} else {
    cat("No valid results found.\n")
}
```

Here, we set the parameters, call the main function, and then print the results for the first three models for a basic validation. If the model runs successfully, you'll see the summary statistics for your stepwise regressions. If there are no results, then a warning message will be output.

In terms of resources for further learning on this subject:

*   **"An Introduction to Statistical Learning" by Gareth James et al.:** This book provides a comprehensive introduction to statistical learning techniques, including regression and variable selection methods. It's excellent for developing a solid understanding of the underlying principles and the practical considerations behind methods like stepwise regression. Pay attention to the sections on model selection and cross-validation.

*   **"Applied Predictive Modeling" by Max Kuhn and Kjell Johnson:** This text offers a more advanced treatment of predictive modeling and model selection techniques. It goes beyond basic methods like stepwise regression and delves into more robust approaches such as regularization and resampling methods that might be more suitable alternatives for actual deployment situations.

*   **"The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman:** This book is a cornerstone resource for advanced statistical learning. While it’s a more theoretical text, it provides in-depth coverage of many methods, with detailed insights into the theory behind them. It includes discussions about model selection that will inform your use of stepwise regression and introduce you to superior alternatives.

Remember, stepwise regression has some serious drawbacks, so be cautious about relying too heavily on it. It is prone to overfitting and can lead to unstable and unreliable models. Using cross-validation and exploring regularization techniques, such as ridge or lasso regression, is very important for robust and generalizable results in most modern real-world applications.

This setup should provide you with a robust start. Let me know if you run into any other issues.
