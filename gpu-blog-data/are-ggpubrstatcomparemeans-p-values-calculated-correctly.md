---
title: "Are `ggpubr::stat_compare_means` p-values calculated correctly?"
date: "2025-01-30"
id: "are-ggpubrstatcomparemeans-p-values-calculated-correctly"
---
The frequent observation of seemingly discrepant p-values between `ggpubr::stat_compare_means` and other statistical packages necessitates a careful examination of its underlying methodology. I've encountered this issue several times during biostatistical analyses and have found that while `ggpubr::stat_compare_means` largely performs correctly, understanding the specifics of its implementation and the potential for user misapplication is crucial. Discrepancies typically arise from the nuances of multiple comparison correction and the default tests employed, not from fundamental flaws in the p-value calculation.

The core functionality of `stat_compare_means` is to conduct statistical hypothesis tests between groups within a plot and annotate the plot with p-values (and optionally significance stars). The function internally leverages various underlying R functions, specifically those within the `stats` package for the statistical tests and, crucially, the `p.adjust` function for multiple comparisons correction. A misunderstanding or misapplication of these internal mechanisms is a common root cause for perceived inaccuracies.

To elaborate, the core test that `stat_compare_means` defaults to is dependent on the user's input. If a user provides a grouping factor with only two levels (e.g., control vs. treatment) and no explicit test is given, it defaults to a Wilcoxon Rank Sum test if the data are not normally distributed or a Student’s t-test otherwise. When more than two levels are presented, it defaults to a Kruskal-Wallis test followed by Dunn's post-hoc test, if `comparisons` are not specified. While these defaults are often reasonable, a user might expect an ANOVA followed by a Tukey post-hoc test, which may lead to different p-values. It's not that `stat_compare_means` is *wrong*, it's just that it might be running a different calculation than the user intended.

The most significant contributing factor to p-value discrepancies I have observed stems from how multiple comparisons are handled. By default, `stat_compare_means` does not perform any correction, meaning that the displayed p-values are raw, unadjusted values from each pairwise test. These raw p-values are the result of individual tests. If multiple tests are conducted, we must be concerned about the increase in Type I error, otherwise known as false positives. In other words, we need to adjust for the number of comparisons being conducted. If we want to avoid the multiple comparisons problem, we need to include an argument telling the function to adjust our p-values. The function makes use of R's `p.adjust` function for multiple comparison corrections (e.g., Bonferroni, Holm, Benjamini-Hochberg). This has been the cause of almost all perceived miscalculations I have come across. The user either did not specify that they wanted multiple comparisons correction to be performed or was not aware that this was not being applied by default.

To illustrate these concepts, consider the following code examples and commentary:

**Example 1: Basic Two-Group Comparison (No Correction)**

```R
library(ggplot2)
library(ggpubr)

set.seed(123)
data <- data.frame(
  group = factor(rep(c("A", "B"), each = 20)),
  value = c(rnorm(20, 5, 1), rnorm(20, 6, 1))
)

p <- ggplot(data, aes(x = group, y = value)) +
  geom_boxplot() +
  stat_compare_means(label = "p")

print(p)
```
In this example, the `stat_compare_means` function is used to display the raw p-value of a two-sample t-test. The output p-value is the direct result of the test without any correction for multiple comparisons since it only involves two groups. This is the default behavior of the function. This p-value can be validated by using `t.test(value ~ group, data = data)` and extracting the p-value. The p-value will be the same.

**Example 2: Multiple Groups, Multiple Comparisons Correction**

```R
library(ggplot2)
library(ggpubr)

set.seed(123)
data_multi <- data.frame(
  group = factor(rep(c("A", "B", "C"), each = 20)),
  value = c(rnorm(20, 5, 1), rnorm(20, 6, 1), rnorm(20, 5.5, 1))
)

p_multi <- ggplot(data_multi, aes(x = group, y = value)) +
  geom_boxplot() +
  stat_compare_means(
    comparisons = list(c("A", "B"), c("A", "C"), c("B", "C")),
    method = "t.test",
    p.adjust.method = "bonferroni",
    label = "p.format"
  )

print(p_multi)
```

Here, we compare three groups (A, B, and C) with pairwise comparisons.  We specify a t-test as the method and, crucially, we specify `"bonferroni"` as the `p.adjust.method`, applying the Bonferroni correction to the raw p-values from the tests. If we were to rerun this code without specifying any adjustment method, the displayed p-values would be smaller since the raw, unadjusted values would be shown. The Bonferroni correction inflates the p-value to account for multiple tests. It is very important to note that if you do not want to account for multiple comparisons, which is sometimes valid depending on the analysis, you should explicitly state that you do *not* want multiple comparisons.

**Example 3: Alternate Nonparametric Test**

```R
library(ggplot2)
library(ggpubr)

set.seed(123)
data_nonpar <- data.frame(
  group = factor(rep(c("A", "B"), each = 20)),
  value = c(rnorm(20, 5, 2), rexp(20, rate = 0.2)) # Simulate non-normal data
)

p_nonpar <- ggplot(data_nonpar, aes(x = group, y = value)) +
  geom_boxplot() +
  stat_compare_means(
    method = "wilcox.test",
    label = "p.format"
  )

print(p_nonpar)
```

In this final example, the data violate the assumption of normality. The `method = "wilcox.test"` argument has explicitly specified that a Wilcoxon rank-sum test should be used instead of the default t-test. Without this argument, the `stat_compare_means` would automatically identify that normality is not met and would conduct a Wilcoxon test anyway. However, it is good practice to be explicit.  This illustrates that `stat_compare_means` also works with nonparametric tests and, just like the t-test, does not correct for multiple comparisons by default.

In conclusion, `ggpubr::stat_compare_means` is calculating p-values correctly given its specified parameters and the underlying statistical functions from R's `stats` package. Perceived discrepancies are almost universally linked to two factors: First, the user's misunderstanding that `stat_compare_means` does not perform a multiple comparison correction by default and needs explicit specification. Second, the user might have expected a different test (ANOVA vs. Kruskal-Wallis) and post-hoc procedure that is not implemented by default. It is not the function that is incorrect, but the user who is either not using it properly, or not fully aware of its default behaviors. To mitigate potential errors, one should always verify that the specified statistical test is correct, be intentional in their decision to correct or not correct for multiple comparisons, and ideally cross-check with output from dedicated statistical functions (like those in the `stats` package) when needed.

For resources on the statistical underpinnings, I would strongly recommend reading materials on hypothesis testing and multiple comparison corrections, specifically materials regarding the Bonferroni, Holm, and Benjamini-Hochberg methods. Textbooks on statistical inference often cover these topics in depth. Documentation for R's built-in `stats` package (specifically, the help files for functions such as `t.test`, `wilcox.test`, `kruskal.test`, `anova`, and `p.adjust`) is also invaluable for understanding the exact calculations that `stat_compare_means` relies on internally. Furthermore, various online resources that discuss the dangers of multiple comparisons are worth looking into. Understanding the underlying statistics is key.
