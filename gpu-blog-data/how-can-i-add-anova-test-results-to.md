---
title: "How can I add ANOVA test results to a box plot using ggpubr in R?"
date: "2025-01-30"
id: "how-can-i-add-anova-test-results-to"
---
The `ggpubr` package in R, while powerful for generating publication-ready plots, doesn't directly integrate ANOVA results onto boxplots through a single function.  My experience working on numerous clinical trial data analyses highlighted this limitation;  `ggpubr` excels at visualization, but statistical annotations require a more multifaceted approach.  Therefore, we need to perform the ANOVA separately and then add the statistically significant comparisons as annotations to the boxplot. This response will detail this process, focusing on clarity and practicality based on my extensive use of these tools.

**1. Clear Explanation of the Process:**

The process involves three distinct stages: conducting the ANOVA test using the `stats` package (or a suitable alternative for more complex designs), extracting the relevant statistical information (p-values for pairwise comparisons), and finally, using `ggpubr`'s annotation features to incorporate this information onto the generated boxplot.  Crucially, the choice of post-hoc test after ANOVA depends on the assumptions of the data and the research question.  For instance, Tukey's HSD is commonly used when comparing all group means, while others like Dunnett's test are suitable for comparing treatment groups to a control.  It is imperative to select an appropriate post-hoc test given the experimental design.

The `ggpubr` package primarily aids in creating visually appealing plots. Its strength lies in simplifying plot generation; however, it does not handle the statistical analysis itself. The statistical results generated from the ANOVA and post-hoc test need to be prepared in a format compatible with `ggpubr`'s annotation capabilities. This usually involves creating a data frame with the group labels and their associated p-values.  These p-values will then be used to generate significance labels which are added to the plot.  The use of a statistically significant threshold, generally 0.05 (alpha level), is crucial for deciding which comparisons should be annotated onto the plot.

**2. Code Examples with Commentary:**

**Example 1: One-way ANOVA with Tukey's HSD**

This example demonstrates a simple one-way ANOVA with Tukey's HSD post-hoc test. This is a common scenario when comparing the means of multiple groups.

```R
# Load necessary libraries
library(ggpubr)
library(rstatix)

# Sample data (replace with your own data)
data <- data.frame(
  group = factor(rep(c("A", "B", "C"), each = 10)),
  value = c(rnorm(10, mean = 10, sd = 2), rnorm(10, mean = 12, sd = 2), rnorm(10, mean = 15, sd = 2))
)

# Perform ANOVA and Tukey's HSD
anova_results <- anova_test(data, value ~ group)
tukey_results <- tukey_hsd(data, value ~ group)

# Create box plot
boxplot <- ggboxplot(data, x = "group", y = "value",
                     color = "group", palette = "jco",
                     add = "jitter")

# Add significance levels to the plot
boxplot + stat_pvalue_manual(
  tukey_results, label = "p.adj.signif",
  y.position = max(data$value) + 0.5,
  step.increase = 0.1
) + labs(title = "One-way ANOVA with Tukey's HSD")

#Print ANOVA results for reference
print(anova_results)

```

This code first performs ANOVA and Tukey's HSD using `rstatix`, a package that simplifies the process. Then it generates the boxplot using `ggpubr`. Finally, `stat_pvalue_manual` adds significant p-values obtained from Tukey's HSD test to the plot, adjusting the y-position for clear display.


**Example 2: Two-way ANOVA with Bonferroni Correction**

Two-way ANOVAs require a different approach to post-hoc testing and p-value adjustments.  In this case, I often utilize a Bonferroni correction to control the family-wise error rate when multiple comparisons are involved.

```R
library(ggpubr)
library(rstatix)

# Sample data (replace with your own data)
data <- data.frame(
  factor1 = factor(rep(rep(c("X", "Y"), each = 10), 2)),
  factor2 = factor(rep(c("A", "B"), each = 20)),
  value = c(rnorm(10, mean = 10, sd = 2), rnorm(10, mean = 12, sd = 2),
            rnorm(10, mean = 11, sd = 2), rnorm(10, mean = 13, sd = 2),
            rnorm(10, mean = 13, sd = 2), rnorm(10, mean = 15, sd = 2),
            rnorm(10, mean = 14, sd = 2), rnorm(10, mean = 16, sd = 2))
)

# Perform two-way ANOVA
anova_results <- anova_test(data, value ~ factor1 * factor2)

#Pairwise comparisons with Bonferroni correction
pwc <- data %>%
  pairwise_t_test(value ~ factor1, paired = FALSE, p.adjust.method = "bonferroni")

#Create boxplot
boxplot<- ggboxplot(data, x = "factor1", y = "value",
                    color = "factor1", palette = "jco",
                    add = "jitter") + facet_wrap(~factor2)

#Add p-values from pairwise comparisons
boxplot + stat_pvalue_manual(pwc, label = "p.adj.signif",
                             y.position = max(data$value) + 0.5,
                             step.increase = 0.1) + labs(title = "Two-way ANOVA with Bonferroni Correction")

#Print ANOVA results for reference
print(anova_results)
```

This example utilizes `pairwise_t_test` from `rstatix` to perform pairwise comparisons for each factor,  incorporating the Bonferroni correction.  The `facet_wrap` function in `ggpubr` is used to create separate boxplots for each level of `factor2`.


**Example 3:  Handling significant interactions in two-way ANOVA**

When a significant interaction is present in a two-way ANOVA, simple pairwise comparisons might be insufficient.  In such cases, you might need to further analyze the data to explore the nature of the interaction.  This might involve creating separate boxplots for each level of one factor and then performing post-hoc tests on each subset.

```R
library(ggpubr)
library(rstatix)

# Sample data (with interaction)
data <- data.frame(
  factor1 = factor(rep(rep(c("X", "Y"), each = 10), 2)),
  factor2 = factor(rep(c("A", "B"), each = 20)),
  value = c(rnorm(10, mean = 10, sd = 2), rnorm(10, mean = 12, sd = 2),
            rnorm(10, mean = 15, sd = 2), rnorm(10, mean = 13, sd = 2),
            rnorm(10, mean = 13, sd = 2), rnorm(10, mean = 11, sd = 2),
            rnorm(10, mean = 16, sd = 2), rnorm(10, mean = 14, sd = 2))
)

#Perform two-way ANOVA
anova_results <- anova_test(data, value ~ factor1 * factor2)

#Analyze interactions (Example: separate analyses for each factor2 level)

#Subset data for factor2 level A
data_A <- subset(data, factor2 == "A")
tukey_A <- tukey_hsd(data_A, value ~ factor1)

#Subset data for factor2 level B
data_B <- subset(data, factor2 == "B")
tukey_B <- tukey_hsd(data_B, value ~ factor1)

#Create box plots with separate annotations
boxplot_A <- ggboxplot(data_A, x = "factor1", y = "value", color = "factor1", palette = "jco", add = "jitter") +
  stat_pvalue_manual(tukey_A, label = "p.adj.signif", y.position = max(data_A$value) + 0.5, step.increase = 0.1) + ggtitle("Factor2 = A")

boxplot_B <- ggboxplot(data_B, x = "factor1", y = "value", color = "factor1", palette = "jco", add = "jitter") +
  stat_pvalue_manual(tukey_B, label = "p.adj.signif", y.position = max(data_B$value) + 0.5, step.increase = 0.1) + ggtitle("Factor2 = B")

#Arrange plots using ggarrange (requires ggpubr)
ggarrange(boxplot_A, boxplot_B, ncol = 2, nrow = 1)

#Print ANOVA results for reference
print(anova_results)
```

This final example demonstrates how to handle significant interactions by performing separate analyses within each level of the interacting factor.  Each subplot then receives its own set of annotations.


**3. Resource Recommendations:**

The R documentation for `ggpubr`, `rstatix`, and the base `stats` package provide comprehensive information on their respective functions.  A good introductory text on statistical methods and R programming will be invaluable in understanding the underlying statistical principles.  Furthermore, consulting specialized statistical texts on ANOVA and post-hoc tests is highly beneficial for more advanced designs and data analysis. Remember to always check the assumptions of your statistical tests before proceeding with the analysis and interpretation of the results.
