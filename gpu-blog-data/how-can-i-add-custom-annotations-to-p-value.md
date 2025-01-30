---
title: "How can I add custom annotations to p-value labels in `ggpubr::stat_compare_means()`?"
date: "2025-01-30"
id: "how-can-i-add-custom-annotations-to-p-value"
---
Statistical comparisons often require visually highlighting specific p-values, not just their numerical representation, to convey nuanced interpretations. While `ggpubr::stat_compare_means()` is excellent for adding statistical annotations, modifying the output text to include custom elements like "n.s." for non-significant results or adding contextual information requires direct manipulation of the underlying ggplot structure. This isn't a feature directly exposed through function parameters, necessitating a layered approach using `ggplot2`'s capabilities. I've encountered this frequently when preparing reports for clinical trials, where each comparison needs a distinct label highlighting its specific context.

The core problem lies in the fact that `stat_compare_means()` generates a series of grobs (graphical objects) that represent the statistical labels and adds them as a layer on the `ggplot` object. These grobs contain the calculated p-values and their positioning information. To add custom text, I have to access the layer, identify the annotation grobs, and replace their text values. This isn't immediately intuitive, as we aren't directly altering `stat_compare_means()` behavior; rather, we modify the resulting ggplot object. This involves understanding how ggplot stores its layers and the objects within them, and a bit of vector-based manipulation. The initial challenge lies in isolating the specific annotation layer because `stat_compare_means` can generate multiple such layers.

The fundamental steps are as follows: First, create your base `ggplot` object with the data you want to analyze. Then, apply `stat_compare_means()` to compute the statistics and add the initial annotation layer. The next step is crucial: I have to extract the layer from the ggplot object using `ggplot_build` and inspect the specific `grobs` object within it. This grob list contains `textGrob` objects that hold the p-values, their coordinates, and their formatting. I can then iterate through these grobs, identify those with the desired annotations, and modify their text attribute.  A key thing to remember is that `stat_compare_means()` applies its annotations based on the comparisons it makes.  To modify the text, you need to understand how the comparison results are stored in the ggplot object.  Finally, once the modified grobs are constructed, the entire annotation layer has to be replaced in the ggplot object using `ggplot_gtable`. This approach offers substantial flexibility, allowing me to introduce complex custom annotations.

Consider this first scenario where I want to label non-significant results with "n.s." rather than just showing the p-value:

```R
library(ggplot2)
library(ggpubr)

# Sample Data
set.seed(42)
data <- data.frame(
    group = rep(c("A", "B", "C"), each = 20),
    value = c(rnorm(20, 5, 1), rnorm(20, 6, 1.5), rnorm(20, 7, 1))
)

# Base ggplot with stat_compare_means
my_plot <- ggplot(data, aes(x = group, y = value)) +
    geom_boxplot() +
    stat_compare_means(comparisons = list(c("A", "B"), c("B", "C"), c("A", "C")))

# Build the ggplot object and extract grobs
gb <- ggplot_build(my_plot)
grobs <- gb$layout$grobs

# Find the annotations layer grob
ann_index <- which(sapply(grobs, function(x) inherits(x, "gTree")) &
                   sapply(grobs, function(x) grepl("stat_compare_means",
                                                  deparse(x$children[[1]]))))


# Access the specific text grobs
grob_list <- grobs[[ann_index]]$children[[1]]$children
text_grobs <- grob_list[sapply(grob_list, function(x) inherits(x, "textGrob"))]

# Iterate through grobs and modify text
for(i in seq_along(text_grobs)) {
    current_text <- text_grobs[[i]]$label
    numeric_val <- as.numeric(gsub("[^0-9\\.]","", current_text)) # Extract p-value
    if (!is.na(numeric_val) && numeric_val > 0.05) {
      text_grobs[[i]]$label <- "n.s."
    }
}

# Replace the old grobs
grobs[[ann_index]]$children[[1]]$children <- grob_list

# Replace the whole layer
gb$layout$grobs <- grobs

# Convert to gtable and plot
my_plot_modified <- ggplot_gtable(gb)
plot(my_plot_modified)

```
In this first example, the `stat_compare_means` initially applies its default annotation, which includes numerical p-values. I then extract the layers using `ggplot_build`, identify the relevant layer containing the annotations, and then refine it to obtain the `textGrobs`. These objects hold the p-values. I then iterate through them, extracting the numerical values, and if the p-value is greater than 0.05, I replace its text with "n.s.". Finally, I replace the manipulated layer back into the original plot for visualization.

In the next instance, I want to add specific annotations that correspond to comparisons; for example, I would like to add text indicating "Treatment vs Control" for one comparison, and "Treatment A vs Treatment B" for the other:
```R
# Sample Data (same as previous example)
set.seed(42)
data <- data.frame(
    group = rep(c("A", "B", "C"), each = 20),
    value = c(rnorm(20, 5, 1), rnorm(20, 6, 1.5), rnorm(20, 7, 1))
)


# Base ggplot with stat_compare_means
my_plot <- ggplot(data, aes(x = group, y = value)) +
    geom_boxplot() +
    stat_compare_means(comparisons = list(c("A", "B"), c("B", "C"), c("A", "C")))

# Build the ggplot object and extract grobs
gb <- ggplot_build(my_plot)
grobs <- gb$layout$grobs


# Find the annotations layer grob
ann_index <- which(sapply(grobs, function(x) inherits(x, "gTree")) &
                   sapply(grobs, function(x) grepl("stat_compare_means",
                                                  deparse(x$children[[1]]))))


# Access the specific text grobs
grob_list <- grobs[[ann_index]]$children[[1]]$children
text_grobs <- grob_list[sapply(grob_list, function(x) inherits(x, "textGrob"))]

# Define custom labels for each comparison
custom_labels <- c("Treatment A vs Control", "Treatment B vs Treatment C", "Treatment A vs Treatment C")


# Match custom labels to comparisons (assumes stat_compare_means order)
for(i in seq_along(text_grobs)) {
    text_grobs[[i]]$label <- paste0(custom_labels[i], "\n p = ", text_grobs[[i]]$label )
}


# Replace the old grobs
grobs[[ann_index]]$children[[1]]$children <- grob_list

# Replace the whole layer
gb$layout$grobs <- grobs


# Convert to gtable and plot
my_plot_modified <- ggplot_gtable(gb)
plot(my_plot_modified)

```
In this version, I have introduced a character vector `custom_labels`. This vector contains annotations for each comparison that corresponds to the order in which they appear in the `comparisons` list within the `stat_compare_means` function. Within the iteration, I replace the original annotation with the custom text combined with the original p-value.

Finally, a third scenario considers dynamically adding annotations based on a user-defined list of p-values and labels:

```R
# Sample Data (same as previous example)
set.seed(42)
data <- data.frame(
    group = rep(c("A", "B", "C"), each = 20),
    value = c(rnorm(20, 5, 1), rnorm(20, 6, 1.5), rnorm(20, 7, 1))
)


# Base ggplot with stat_compare_means
my_plot <- ggplot(data, aes(x = group, y = value)) +
    geom_boxplot() +
    stat_compare_means(comparisons = list(c("A", "B"), c("B", "C"), c("A", "C")))

# Build the ggplot object and extract grobs
gb <- ggplot_build(my_plot)
grobs <- gb$layout$grobs


# Find the annotations layer grob
ann_index <- which(sapply(grobs, function(x) inherits(x, "gTree")) &
                   sapply(grobs, function(x) grepl("stat_compare_means",
                                                  deparse(x$children[[1]]))))

# Access the specific text grobs
grob_list <- grobs[[ann_index]]$children[[1]]$children
text_grobs <- grob_list[sapply(grob_list, function(x) inherits(x, "textGrob"))]

# User Defined p-values and annotations
user_pvals_and_labels <- list(
    list(pval = 0.045, annotation = "Statistically significant (p < 0.05)"),
    list(pval = 0.07, annotation = "Borderline significance"),
    list(pval = 0.002, annotation = "Highly significant (p < 0.01)")
)

# Function to apply user-defined annotations
apply_annotation <- function(text_grobs, user_defined_list){
    for(i in seq_along(text_grobs)) {
          current_text <- text_grobs[[i]]$label
          numeric_val <- as.numeric(gsub("[^0-9\\.]","", current_text))
      for (entry in user_defined_list) {
              if (!is.na(numeric_val) &&  abs(numeric_val - entry$pval)  < 0.001 ) { # Compare floats using abs diff
                      text_grobs[[i]]$label <- paste0(text_grobs[[i]]$label, "\n" ,entry$annotation)
                  break;
              }
          }
      }
  return(text_grobs)
}

modified_text_grobs <- apply_annotation(text_grobs, user_pvals_and_labels)

# Replace the old grobs
grobs[[ann_index]]$children[[1]]$children <- modified_text_grobs

# Replace the whole layer
gb$layout$grobs <- grobs


# Convert to gtable and plot
my_plot_modified <- ggplot_gtable(gb)
plot(my_plot_modified)
```

Here, I have introduced `user_pvals_and_labels`, a list of lists, which associate p-values with specific annotations. The function `apply_annotation` iterates through the annotation grobs extracted from `ggplot`, checks if any p-value extracted from the labels matches a value in `user_pvals_and_labels`. If found, the new annotation is appended to the label. This approach offers a dynamic way to customize annotations based on context.  I used `abs(numeric_val - entry$pval) < 0.001` because directly comparing floating-point numbers can be unreliable due to precision limitations.

For further study, I recommend focusing on the documentation and examples for `ggplot2` and its internal structure, specifically `ggplot_build`, `gTree`, `textGrob`, `ggplot_gtable`.  Understanding the concepts of `grobs` and layers, will provide a very good base. Further exploration into the `grid` package which forms the base of `ggplot2`'s graphics engine will also be beneficial. Finally, studying the structure of the output generated by `stat_compare_means` will prove valuable in understanding how to target specific elements for customization.
