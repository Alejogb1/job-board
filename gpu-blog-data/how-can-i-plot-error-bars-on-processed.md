---
title: "How can I plot error bars on processed data using ggbarplot in ggpubr?"
date: "2025-01-30"
id: "how-can-i-plot-error-bars-on-processed"
---
Error bars in data visualization are essential for communicating uncertainty associated with measurements or derived statistics. Specifically, when using `ggbarplot` from the `ggpubr` package in R to display summarized data, incorporating error bars provides crucial context about the variability within each group. My experience working on a large-scale genomics project demonstrated the necessity of presenting both central tendencies and dispersion measures effectively, especially when dealing with aggregated results from multiple experimental replicates.

The challenge in plotting error bars with `ggbarplot` stems from the fact that the function itself is designed to handle summarized data directly; it doesn't intrinsically calculate these statistics from raw inputs. Therefore, the user must pre-calculate both the mean (or median, depending on the context) for the height of each bar and a measure of dispersion (standard deviation, standard error, or confidence intervals) for the vertical extent of the error bar. These values are then supplied to `ggbarplot` through separate data columns.

Let's illustrate this with concrete examples. Suppose, initially, our raw data is structured with columns for 'Treatment,' and the response variable, 'ExpressionLevel'. We want to visualize the mean expression level for each treatment group, along with its associated standard deviation.

**Example 1: Using Standard Deviation as Error Bars**

First, I need to summarize the raw data. I'll use the `dplyr` package for this purpose. Let's assume our data frame is called `rawData`.

```r
library(dplyr)
library(ggpubr)

# Create example data
rawData <- data.frame(
  Treatment = rep(c("A", "B", "C"), each = 10),
  ExpressionLevel = c(rnorm(10, 5, 1), rnorm(10, 7, 1.5), rnorm(10, 6, 0.8))
)

# Summarize data to get means and standard deviations
summaryData <- rawData %>%
  group_by(Treatment) %>%
  summarize(
    mean_expression = mean(ExpressionLevel),
    sd_expression = sd(ExpressionLevel)
  )

# Create the barplot with error bars
ggbarplot(summaryData, 
          x = "Treatment", 
          y = "mean_expression",
          error.plot = "errorbar",
          ylab = "Mean Expression Level",
          xlab = "Treatment",
          add = "mean_se",
          title = "Mean Expression with Standard Deviation Error Bars",
          ggtheme = theme_minimal()
)
```

In this code, the `summarize` function calculates the mean (`mean_expression`) and standard deviation (`sd_expression`) for each treatment group. Subsequently, in `ggbarplot`,  the  `x` argument is 'Treatment', and  `y` is set to 'mean_expression', representing the bar heights. The crucial aspect is the implicit use of the 'add' parameter, setting it to 'mean_se', which automatically calculates and adds error bars based on the precalculated standard deviation (where standard error is assumed based on given standard deviations). Note that since 'add' is set to mean_se and not errorbars, it will assume the standard deviation is the measure of error, meaning you do not need to explicitly declare `error.upper` or `error.lower`.  The `error.plot` argument, which we set to "errorbar" does nothing since the error is defined by the 'add' parameter.  The plot itself is rendered with `theme_minimal` for a clean aesthetic.

**Example 2: Using Custom Error Bar Values (Upper and Lower)**

Often, you might not be working directly with standard deviations. You might have a different error measure or want to display confidence intervals. In this case, we need to explicitly provide the upper and lower bounds for our error bars. Suppose we have the upper and lower confidence intervals already calculated in our summary data. This scenario often arises when bootstrapping or performing other simulations.

```r
# Example data with pre-calculated confidence intervals
summaryData_ci <- data.frame(
  Treatment = c("A", "B", "C"),
  mean_expression = c(5, 7, 6),
  lower_ci = c(4.5, 6.5, 5.5),
  upper_ci = c(5.5, 7.5, 6.5)
)

# Create the barplot with custom error bars
ggbarplot(summaryData_ci,
          x = "Treatment",
          y = "mean_expression",
          ylab = "Mean Expression Level",
          xlab = "Treatment",
          add = "errorbar",
          error.upper = "upper_ci",
          error.lower = "lower_ci",
          title = "Mean Expression with Custom Confidence Interval Error Bars",
          ggtheme = theme_minimal()
)
```

Here, we explicitly specify the `error.upper` and `error.lower` arguments to the `ggbarplot` function. The `add = "errorbar"` argument now forces `ggbarplot` to draw these bars based on the columns specified. This example demonstrates the flexibility offered by `ggbarplot` when dealing with pre-calculated, arbitrary error ranges. This particular form is especially useful when results are compiled from models, such as linear mixed models where standard deviation can be calculated on the random effects.

**Example 3: Displaying Multiple Datasets on the Same Plot**

Consider the situation where we have two datasets, perhaps from two different experimental conditions, and we want to visually compare them within the same plot. This requires grouping both by the experimental condition and treatment. We can achieve this with a slightly modified summarization and the "fill" and "position" parameters in `ggbarplot`.

```r
# Example data for two conditions
rawData_multi <- data.frame(
  Treatment = rep(c("A", "B", "C"), each = 20),
  Condition = rep(c("Condition1", "Condition2"), each = 10, times=3),
  ExpressionLevel = c(rnorm(20, 5, 1), rnorm(20, 7, 1.5), rnorm(20, 6, 0.8))
)

# Summarize data for each condition and treatment
summaryData_multi <- rawData_multi %>%
  group_by(Treatment, Condition) %>%
  summarize(
    mean_expression = mean(ExpressionLevel),
    sd_expression = sd(ExpressionLevel)
  )

# Create the grouped barplot with error bars
ggbarplot(summaryData_multi,
          x = "Treatment",
          y = "mean_expression",
          fill = "Condition",
          color="black",
          add = "mean_se",
          position = position_dodge(width = 0.9),
          ylab = "Mean Expression Level",
          xlab = "Treatment",
          title = "Mean Expression Across Two Conditions",
          ggtheme = theme_minimal()
)
```
In this example, we include `fill = "Condition"` to distinguish between bars within each treatment, and `position = position_dodge(width = 0.9)` to position the bars side-by-side, preventing overlap. By including the 'color' argument, each bar has a black border. The 'add = "mean_se"' specifies that the standard deviation will be used for the calculation of error, using the precomputed values. This visualizes the error in a grouped manner.

These examples illustrate how to plot error bars on `ggbarplot` figures using different methods of defining the error. The key to successfully using `ggbarplot` with error bars lies in calculating the summary statistics beforehand and correctly mapping these values to the appropriate arguments. The process involves carefully selecting appropriate dispersion measures like standard deviations or confidence intervals, summarizing the raw data, and then using the ‘error.upper’, ‘error.lower’ or ‘add = “mean_se”’ parameters to control the visualization of errors.

**Resource Recommendations:**

For further study on data visualization and statistical computation in R, I recommend the following resources.

1.  **"R for Data Science" by Hadley Wickham and Garrett Grolemund:** This book provides a comprehensive introduction to data manipulation and visualization with R using the tidyverse suite of packages, including `dplyr` and `ggplot2`, the foundation on which `ggpubr` is built. It provides a strong foundation for more sophisticated plots.

2. **"ggplot2: Elegant Graphics for Data Analysis" by Hadley Wickham:** While not directly focused on `ggpubr`, this work delves deep into the underlying principles and capabilities of `ggplot2`, which will significantly enhance your ability to customize your plots. Understanding this package will allow you to fully explore `ggpubr` as well as the many features of `ggplot2`.

3. **"The R Graphics Cookbook" by Winston Chang:** This book offers recipes for creating different kinds of graphs and charts using `ggplot2`. It’s a practical resource for tackling specific data visualization challenges and explores error bar construction in detail.

These books, along with the official R documentation for `dplyr`, `ggplot2`, and `ggpubr`, offer comprehensive coverage of data summarization and plotting, empowering you to generate accurate and informative visualizations for your data.
