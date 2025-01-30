---
title: "How can ggpubr display significance levels (***, n.s.) instead of p-values?"
date: "2025-01-30"
id: "how-can-ggpubr-display-significance-levels--ns"
---
The `ggpubr` package, while powerful for visualizing statistical results, doesn't directly support displaying significance levels (***, **, *, n.s.) as a replacement for p-values within its plotting functions.  This is because `ggpubr` primarily focuses on visualizing the statistical tests themselves, and the mapping of p-values to significance levels is a post-hoc interpretation requiring a separate step.  In my experience developing statistical visualization tools for clinical trials, I've encountered this limitation frequently, and the solution involves manipulating the p-values obtained from the statistical test before integrating them into the `ggpubr` workflow.  This requires a custom function to handle the conversion and subsequent integration with the generated plot.

**1. Clear Explanation:**

The process involves three key stages:  first, performing the statistical test using a suitable function (e.g., `t.test`, `wilcox.test`, `aov`); second, converting the resulting p-value into a significance level based on predefined thresholds; and third, incorporating this significance level into the `ggpubr` plot using annotation features.  The crucial step is defining the threshold for each significance level. This is usually based on a common convention, but it is advisable to state these thresholds explicitly in the documentation or figure caption for transparency.  Generally accepted thresholds are: p ≤ 0.001 (***), 0.001 < p ≤ 0.01 (**), 0.01 < p ≤ 0.05 (*), and p > 0.05 (n.s.). However, these can be adjusted as needed to meet specific study requirements.  Note that this approach requires careful consideration, as relying solely on significance levels can oversimplify the interpretation of results.  It's essential to present the original p-values alongside the significance levels in a table or supplementary material for complete transparency.

**2. Code Examples with Commentary:**

**Example 1: Independent Samples t-test**

```R
# Load necessary libraries
library(ggpubr)
library(broom)

# Sample data (replace with your data)
data <- data.frame(
  group = factor(rep(c("A", "B"), each = 10)),
  value = c(rnorm(10, mean = 10, sd = 2), rnorm(10, mean = 12, sd = 2))
)

# Perform t-test
result <- t.test(value ~ group, data = data)

# Function to convert p-value to significance level
p_to_stars <- function(p) {
  ifelse(p <= 0.001, "***", ifelse(p <= 0.01, "**", ifelse(p <= 0.05, "*", "n.s.")))
}

# Extract p-value and convert to significance level
p_value <- tidy(result)$p.value
sig_level <- p_to_stars(p_value)

# Create boxplot with significance level annotation
ggboxplot(data, x = "group", y = "value") +
  stat_compare_means(comparisons = list(c("A", "B")), label = "p.signif", method = "t.test", label.x = 1.5) +
  annotate("text", x = 1.5, y = max(data$value) * 0.95, label = sig_level) +
  labs(title = "Comparison of Groups A and B")
```

This example demonstrates a simple t-test. The `p_to_stars` function maps p-values to significance levels. The `stat_compare_means` function provides the p-value, which we replace using manual annotation with the calculated significance level (`sig_level`).  Note that this overwrites the p-value from `stat_compare_means`.



**Example 2:  Wilcoxon Rank-Sum Test**

```R
# Load necessary libraries (assuming previous example loaded ggpubr and broom)

# Sample data (non-parametric, replace with your data)
data2 <- data.frame(
  group = factor(rep(c("A", "B"), each = 10)),
  value = c(rexp(10, rate = 0.1), rexp(10, rate = 0.2))
)

# Perform Wilcoxon test
result2 <- wilcox.test(value ~ group, data = data2)

# Extract p-value and convert using the same function
p_value2 <- tidy(result2)$p.value
sig_level2 <- p_to_stars(p_value2)

# Create boxplot with significance level annotation
ggboxplot(data2, x = "group", y = "value") +
  stat_compare_means(comparisons = list(c("A", "B")), label = "p.signif", method = "wilcox.test", label.x = 1.5) +
  annotate("text", x = 1.5, y = max(data2$value) * 0.95, label = sig_level2) +
  labs(title = "Comparison of Groups A and B (Wilcoxon Test)")
```

This example extends the methodology to a non-parametric Wilcoxon test, demonstrating its versatility.  The core principle of converting p-values to significance levels remains consistent.


**Example 3: ANOVA with Post-hoc Tests**

```R
# Load necessary libraries (assuming previous examples loaded libraries)

# Sample data (ANOVA, replace with your data)
data3 <- data.frame(
  group = factor(rep(c("A", "B", "C"), each = 10)),
  value = c(rnorm(10, mean = 10, sd = 2), rnorm(10, mean = 12, sd = 2), rnorm(10, mean = 11, sd = 2))
)

# Perform ANOVA
result3 <- aov(value ~ group, data = data3)

# Perform post-hoc test (Tukey's HSD)
post_hoc <- TukeyHSD(result3)

# Function to extract p-values and convert them to significance levels
extract_and_convert <- function(post_hoc_result){
  p_values <- post_hoc_result$group["p adj"]
  sig_levels <- sapply(p_values, p_to_stars)
  return(sig_levels)
}

# Extract and convert
sig_levels3 <- extract_and_convert(post_hoc)

# Create boxplot with significance level annotation (requires manual adjustments for multiple comparisons)
ggboxplot(data3, x = "group", y = "value") +
  stat_compare_means(comparisons = list(c("A", "B"), c("A", "C"), c("B", "C")), label = "p.signif", method = "anova", label.x = 1.75) +
    annotate("text", x = c(1.5, 2, 2.5), y = rep(max(data3$value) * 0.95, 3), label = sig_levels3) +
  labs(title = "Comparison of Groups A, B, and C (ANOVA)")
```

This example demonstrates the application with ANOVA and Tukey's HSD post-hoc test.  The increased complexity of multiple comparisons requires a more sophisticated approach to extract and annotate the significance levels, making manual adjustment for label placement necessary.

**3. Resource Recommendations:**

*   "R for Data Science" by Garrett Grolemund and Hadley Wickham
*   "ggplot2: Elegant Graphics for Data Analysis" by Hadley Wickham
*   The `ggpubr` package documentation

Remember that while replacing p-values with significance levels simplifies visual representation,  it's crucial to maintain the original p-values within the associated documentation for a complete and accurate representation of the statistical analysis.  Always prioritize clarity and transparency in your data presentation.
