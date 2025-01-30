---
title: "Why do Limma moderated t-tests and `stat_compare_means` produce different results?"
date: "2025-01-30"
id: "why-do-limma-moderated-t-tests-and-statcomparemeans-produce"
---
In my experience, discrepancies between Limma's moderated t-tests and the results from `stat_compare_means` in R's `ggpubr` package often stem from fundamental differences in their underlying statistical models and the handling of variance. Specifically, Limma employs an empirical Bayes method to shrink sample variances toward a common value, while `stat_compare_means` typically performs standard, unmoderated t-tests on a per-comparison basis. This difference in variance estimation is the primary source of the often-observed divergence in p-values and significance assessments.

Let's examine the specific mechanisms involved. Limma, primarily designed for analyzing gene expression data, recognizes that in experiments with a limited number of replicates, individual sample variances can be highly unstable and misleading. Instead of relying solely on these potentially volatile estimates, Limma leverages information across all genes to compute a *pooled variance estimate*. This process, called 'moderation,' borrows strength from the entire dataset to improve the reliability of variance estimation, particularly for genes with few samples.

The core principle of Limma's approach is to treat each gene’s individual variance as a noisy estimate of a common underlying variance. It then shrinks these estimates towards a common average, employing empirical Bayes methods. The extent of shrinkage is inversely proportional to the stability of each variance: highly variable variance estimates are shrunk more aggressively than relatively stable ones. This process stabilizes the variance estimates, and consequently, the associated t-statistics and p-values, especially for cases with small sample sizes. The `eBayes` function in Limma is the engine for this moderation, resulting in moderated t-statistics that are more robust.

On the other hand, `stat_compare_means`, part of the `ggpubr` package, defaults to performing standard t-tests for group comparisons, where variance is estimated independently for each comparison. It does not typically incorporate any prior information or moderation procedure. These standard t-tests use the observed variance in each group to calculate a t-statistic, and subsequently, a p-value. When there are few replicates, the resulting per-comparison variances can be quite variable.

Therefore, in a scenario with high-throughput data, where many comparisons are made with few replicates within each group, Limma's moderated t-tests are designed to provide more reliable and consistent results than the raw t-tests produced by `stat_compare_means`. The former is effectively borrowing information across the dataset to achieve a more stable and less variable assessment of differential expression, while the latter treats each comparison as an independent, isolated experiment.

Now, let’s look at some examples demonstrating how these methods diverge.

**Code Example 1: Data Generation and Analysis**

```R
library(limma)
library(ggpubr)

# Simulate gene expression data with 3 replicates per group for 10 genes
set.seed(123)
n_genes <- 10
n_replicates <- 3
group1 <- rnorm(n_genes * n_replicates, mean = 0, sd = 1)
group2 <- rnorm(n_genes * n_replicates, mean = 0.5, sd = 1) # some genes are differentially expressed

expression_data <- matrix(c(group1, group2), ncol = 2 * n_replicates, byrow = FALSE)
colnames(expression_data) <- c(paste0("G1_rep", 1:n_replicates), paste0("G2_rep", 1:n_replicates))
rownames(expression_data) <- paste0("gene", 1:n_genes)

# Create a design matrix for Limma
design <- matrix(c(rep(1, 2 * n_replicates), rep(c(0, 1), each = n_replicates)), ncol = 2)
colnames(design) <- c("Intercept", "Group2")

# Perform Limma analysis
fit <- lmFit(expression_data, design)
fit <- eBayes(fit)
limma_results <- topTable(fit, coef = 2, number = Inf)

# Convert to long format for stat_compare_means
long_data <- data.frame(expression = c(expression_data),
                         gene = rep(rownames(expression_data), each = 2 * n_replicates),
                         group = rep(c("G1", "G2"), each = n_replicates, times = n_genes))

# Perform stat_compare_means analysis
stat_results <- compare_means(expression ~ group, data = long_data, group.by = "gene", method = "t.test")
stat_results <- stat_results[, c("gene", "p")]

# Merge and compare p-values
merged_results <- merge(limma_results[, c("ID", "P.Value")], stat_results, by.x = "ID", by.y = "gene")
print(head(merged_results))
```
**Commentary:**
This example sets up a simulation of gene expression data. I am creating data where some genes are differentially expressed (with a mean difference between groups). It then demonstrates how to use `limma` and `stat_compare_means`. The crucial step is the `eBayes` call within the Limma analysis, which performs the empirical Bayes moderation. The p-values from Limma, as shown in the printed output, are typically smaller (more significant) for truly differentially expressed genes and more consistent across genes compared to those calculated by t.tests within `stat_compare_means`, due to variance stabilization. The `head` function shows the discrepancy in the p-values.

**Code Example 2: Impact of Limited Replicates**

```R
#Simulate data with fewer replicates
set.seed(456)
n_replicates_small <- 2 # Reduced replicates
group1_small <- rnorm(n_genes * n_replicates_small, mean = 0, sd = 1)
group2_small <- rnorm(n_genes * n_replicates_small, mean = 0.5, sd = 1)
expression_data_small <- matrix(c(group1_small, group2_small), ncol = 2 * n_replicates_small, byrow = FALSE)
colnames(expression_data_small) <- c(paste0("G1_rep", 1:n_replicates_small), paste0("G2_rep", 1:n_replicates_small))
rownames(expression_data_small) <- paste0("gene", 1:n_genes)

# Limma analysis with smaller sample size
design_small <- matrix(c(rep(1, 2 * n_replicates_small), rep(c(0, 1), each = n_replicates_small)), ncol = 2)
colnames(design_small) <- c("Intercept", "Group2")
fit_small <- lmFit(expression_data_small, design_small)
fit_small <- eBayes(fit_small)
limma_results_small <- topTable(fit_small, coef = 2, number = Inf)

# Convert to long format for stat_compare_means
long_data_small <- data.frame(expression = c(expression_data_small),
                         gene = rep(rownames(expression_data_small), each = 2 * n_replicates_small),
                         group = rep(c("G1", "G2"), each = n_replicates_small, times = n_genes))

# Perform stat_compare_means analysis
stat_results_small <- compare_means(expression ~ group, data = long_data_small, group.by = "gene", method = "t.test")
stat_results_small <- stat_results_small[, c("gene", "p")]

# Merge and compare p-values for smaller sample size data
merged_results_small <- merge(limma_results_small[, c("ID", "P.Value")], stat_results_small, by.x = "ID", by.y = "gene")
print(head(merged_results_small))
```
**Commentary:**
This code demonstrates the impact of reducing the number of replicates. By decreasing the replicates to just two per group, the instability in per-comparison variance increases. The difference in p-values generated from Limma (using variance moderation) and standard t-tests becomes more pronounced. The standard t-tests will be less reliable (more variable) due to lack of information. This highlights the advantage of Limma, as it will reduce variance estimates across the genes providing more stable p-values.

**Code Example 3: Large Dataset Illustration**

```R
set.seed(789)
n_genes_large <- 1000
n_replicates_large <- 4
group1_large <- rnorm(n_genes_large * n_replicates_large, mean = 0, sd = 1)
group2_large <- rnorm(n_genes_large * n_replicates_large, mean = 0.3, sd = 1) # smaller mean difference
expression_data_large <- matrix(c(group1_large, group2_large), ncol = 2 * n_replicates_large, byrow = FALSE)
colnames(expression_data_large) <- c(paste0("G1_rep", 1:n_replicates_large), paste0("G2_rep", 1:n_replicates_large))
rownames(expression_data_large) <- paste0("gene", 1:n_genes_large)

# Limma analysis for large dataset
design_large <- matrix(c(rep(1, 2 * n_replicates_large), rep(c(0, 1), each = n_replicates_large)), ncol = 2)
colnames(design_large) <- c("Intercept", "Group2")
fit_large <- lmFit(expression_data_large, design_large)
fit_large <- eBayes(fit_large)
limma_results_large <- topTable(fit_large, coef = 2, number = Inf)

# Convert to long format for stat_compare_means
long_data_large <- data.frame(expression = c(expression_data_large),
                           gene = rep(rownames(expression_data_large), each = 2 * n_replicates_large),
                           group = rep(c("G1", "G2"), each = n_replicates_large, times = n_genes_large))

# Perform stat_compare_means analysis
stat_results_large <- compare_means(expression ~ group, data = long_data_large, group.by = "gene", method = "t.test")
stat_results_large <- stat_results_large[, c("gene", "p")]


# Merge and compare p-values for large data
merged_results_large <- merge(limma_results_large[, c("ID", "P.Value")], stat_results_large, by.x = "ID", by.y = "gene")
head(merged_results_large)

#calculate the correlation between limma p-values and stat p-values
print(cor(merged_results_large$P.Value, merged_results_large$p))
```
**Commentary:**
This example demonstrates how, with a large dataset of 1000 genes, the p-values generated by both tests are somewhat correlated, as shown by the output of the correlation call. However, the correlation is not perfect and p-values will still show some variation between the two methods. The difference stems from the fact that Limma is performing moderation even with four samples per group, while standard t-tests perform a per-comparison test, and this leads to a difference in the resulting p-values. The moderation aspect of Limma is designed to reduce the impact of noisy variances, especially when working with genomic data, where variations in expression could stem from several biological or technical factors.

For further resources, I recommend exploring textbooks and monographs focused on linear models and microarray data analysis. Publications from the core Bioconductor team (including the original Limma paper and those discussing empirical Bayes methods) offer foundational knowledge. Further, consulting comprehensive guides covering statistical techniques for omics data, are useful for a deeper understanding of these statistical concepts. These will provide a detailed exposition of the theoretical underpinnings of these methods.
