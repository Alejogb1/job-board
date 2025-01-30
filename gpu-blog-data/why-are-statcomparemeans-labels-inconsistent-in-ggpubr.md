---
title: "Why are stat_compare_means labels inconsistent in ggpubr?"
date: "2025-01-30"
id: "why-are-statcomparemeans-labels-inconsistent-in-ggpubr"
---
The inconsistency in `stat_compare_means` labels within the `ggpubr` R package stems primarily from its reliance on underlying statistical test assumptions and the various label formatting options available. In my experience debugging numerous visualizations for biomedical research, I've frequently encountered these seemingly unpredictable label positions and textual outputs, requiring a deep understanding of the function's mechanics.

At its core, `stat_compare_means` automatically determines which statistical test to perform based on the data characteristics and the user-specified method. This automated selection, while convenient, contributes to label inconsistencies when the underlying test's output varies in format. For instance, a t-test generates a p-value directly, which is straightforward to display. However, a Wilcoxon test may provide an adjusted p-value with a different level of significance or require a different string representation.

The label position is another key source of inconsistency. `stat_compare_means` defaults to placing labels above brackets connecting the comparison groups. This simplistic approach often leads to overlapping labels, labels that obscure data points, or labels that are visually disjointed when the y-axis scaling varies significantly. The built-in label positioning algorithm does not robustly handle datasets with substantial differences in value ranges. It frequently positions labels based on a simple addition to the maximum value within the range, which is unreliable when distributions are skewed or data exhibits outliers. Moreover, when multiple comparisons are drawn, these overlapping labels become increasingly hard to read and understand. Finally, user-defined positioning is often not an intuitive process because of limitations in direct manipulation via numeric coordinates.

The `label` argument, although versatile, can also create unexpected outputs if not utilized carefully. While it supports options like "p.signif" to denote significance levels (e.g., * for p < 0.05), "p.format" to present numerical values, and custom strings, incorrect use can lead to ambiguity or misrepresentation of the statistical results. Furthermore, formatting options passed to the `label` argument can interact unpredictably with the base formatting of `stat_compare_means`. There's a need for careful consideration and testing when combining these formatting features.

Let's examine some specific cases to illustrate these points. First, consider the simplest case, a two-group comparison using a t-test.

```R
library(ggplot2)
library(ggpubr)

# Create sample data
set.seed(42)
data_t_test <- data.frame(
    group = factor(rep(c("A", "B"), each = 20)),
    value = c(rnorm(20, mean = 5, sd = 1), rnorm(20, mean = 6, sd = 1.2))
)

# Basic t-test comparison
ggplot(data_t_test, aes(x = group, y = value)) +
    geom_boxplot() +
    stat_compare_means(method = "t.test") +
    labs(title="Two-Sample t-test")
```
In this example, `stat_compare_means` will default to a Welch t-test, given that no assumptions were made about variances. The label will display the p-value, which is consistent across multiple runs of this test on the same data. The placement is above the brackets, typically reasonable.

However, this changes with a non-parametric test like the Wilcoxon test, where the label shows a different p-value format, and the behavior starts to deviate when we have multiple comparisons. Consider the following case using data that do not satisfy assumptions of the previous example:

```R
# Sample data with non-normal distribution
set.seed(42)
data_wilcox <- data.frame(
    group = factor(rep(c("X", "Y", "Z"), each = 20)),
    value = c(rexp(20, rate = 1), rexp(20, rate = 0.7), rexp(20, rate = 0.8))
)

# Wilcoxon test with multiple comparisons
ggplot(data_wilcox, aes(x = group, y = value)) +
    geom_boxplot() +
    stat_compare_means(method = "wilcox.test", comparisons = list(c("X", "Y"), c("X", "Z"), c("Y","Z"))) +
    labs(title="Multiple Wilcoxon Comparisons")
```
Here, `stat_compare_means` will perform Wilcoxon rank-sum tests. The p-values are likely to have a different textual format from the previous example, and the labels might overlap due to the relatively high number of comparisons and the default placement logic.  The numerical outputs will often be formatted differently to accommodate the scientific notation if the p-values are too small, and there might be situations where some comparisons have significant p-values but not in the same order of magnitude as the other comparisons.

Finally, changing label formats can exacerbate the issue. Let us use the first data example with the Welch-t test and add custom formats:

```R
# Custom label formats
ggplot(data_t_test, aes(x = group, y = value)) +
    geom_boxplot() +
    stat_compare_means(method = "t.test", label = "p.format", label.x=1.5) +
    labs(title="Custom Format t-test")
```
Here, we are forcing the label to use numerical p-values and shift it using `label.x`. Although the format is now consistent (numerical) and positioned at `label.x`, one can imagine scenarios where custom values will overlap in multi-comparison scenarios.

To mitigate these inconsistencies, I advise employing the following strategies based on my own work: First, carefully evaluate if default settings are appropriate for the data distribution. If data strongly violate parametric assumptions, non-parametric approaches are essential and user needs to account for the variations in outputs of these tests. Second, manual adjustment of the `y` position for the labels using `label.y` should be the preferred way, as opposed to relying on defaults. Third, if dealing with many comparisons, the adjustment of the x-position can also be beneficial using `label.x` argument, and using `vjust` or `hjust`. Fourth, if using custom label, consider pre-formatting the text outside of the plot for consistency.

For additional insight and guidance, I recommend consulting the official `ggpubr` documentation. The documentation provides detailed descriptions of each function parameter, which is helpful for understanding their impact on the output and label placement. Also the user can inspect the code of the underlying function `stat_compare_means` in R to understand how the p-values are derived and formatted. Furthermore, the book “ggplot2 Elegant Graphics for Data Analysis” is a useful companion, because it demonstrates how the ggplot2 grammar of graphics works and how we can control every detail of the plots. Finally, exploring examples of plot modifications presented on public forums can be a valuable learning resource, but one must be careful of the solutions that are not reproducible or generalizable. Applying these practices has enabled me to create clear, informative visualizations while mitigating common `stat_compare_means` label inconsistencies.
