---
title: "How can I display values and sample sizes in a balloon plot using ggballoonplot in R?"
date: "2025-01-30"
id: "how-can-i-display-values-and-sample-sizes"
---
The `ggballoonplot` function, while visually appealing for displaying data distributions, lacks a direct, built-in mechanism for simultaneously presenting both the value and sample size within each balloon.  My experience working with large epidemiological datasets necessitates this functionality for clear data interpretation; simply representing magnitude through balloon size can be ambiguous without explicit numerical annotation. Therefore, achieving this requires a layered approach, leveraging `ggballoonplot`'s base functionality and augmenting it with `ggplot2`'s text annotation capabilities.

**1. Clear Explanation:**

The core strategy involves generating the balloon plot using `ggballoonplot`, then layering text geoms onto the plot to display both the value and the sample size.  We need to extract the relevant data (value and count) from the input data frame to properly position the text labels.  Precise positioning requires careful consideration; ideally, the text labels should be clearly visible without overlapping balloons or obscuring crucial visual elements.  This often involves adjusting horizontal and vertical offsets using `hjust` and `vjust` parameters within the `geom_text` function.  Furthermore,  the font size and color should be selected to ensure readability without visual clutter. In instances with many overlapping balloons,  consider using a smaller font or a more transparent background for the text labels.  For very dense plots, alternative strategies such as interactive plots might be preferable.

**2. Code Examples with Commentary:**

**Example 1: Basic Annotation**

This example demonstrates the fundamental approach: creating the balloon plot and overlaying the value and sample size as text.

```R
library(ggballoonplot)
library(ggplot2)

# Sample data
data <- data.frame(
  group = factor(rep(LETTERS[1:3], each = 10)),
  value = rnorm(30, mean = 10, sd = 2),
  count = sample(5:20, 30, replace = TRUE)
)

# Create the balloon plot
plot <- ggballoonplot(data, x = "group", y = "value", size = "count",
                       fill = "group", show.legend = FALSE) +
  scale_size(range = c(2, 10)) + # Adjust balloon size range for better visualization

  # Add text annotations for value
  geom_text(aes(label = round(value, 1)), position = position_nudge(y = 2), size = 3, color = "black") +

  # Add text annotations for count (sample size)
  geom_text(aes(label = paste0("n=", count)), position = position_nudge(y = -2), size = 3, color = "black")

print(plot)

```

This code first creates a sample dataset.  `ggballoonplot` generates the plot, and `geom_text` adds the value and sample size as text labels.  `position_nudge` slightly offsets the labels vertically for clarity, and `size` and `color` control the text appearance.


**Example 2: Handling Overlapping Labels**

This example demonstrates a more sophisticated approach that addresses potential label overlap. It uses `ggrepel` to avoid overlapping labels, enhancing readability.

```R
library(ggballoonplot)
library(ggplot2)
library(ggrepel)

#Using the same data as Example 1

plot <- ggballoonplot(data, x = "group", y = "value", size = "count",
                       fill = "group", show.legend = FALSE) +
  scale_size(range = c(2, 10)) +

  geom_text_repel(aes(label = round(value, 1)), size = 3, color = "black",  box.padding = 0.5, segment.size = 0.2) +
  geom_text_repel(aes(label = paste0("n=", count)), size = 3, color = "black", nudge_y = -2, box.padding = 0.5, segment.size = 0.2)

print(plot)
```

Here, `geom_text_repel` from the `ggrepel` package is used.  This function intelligently positions text labels to minimize overlap, improving readability, especially in denser plots.  `box.padding` and `segment.size` further fine-tune the label placement and connecting lines.

**Example 3:  Conditional Formatting**

This example incorporates conditional formatting to highlight specific data points based on a threshold.

```R
library(ggballoonplot)
library(ggplot2)
library(ggrepel)

# Sample data with a threshold
data$threshold_exceeded <- data$value > 12


plot <- ggballoonplot(data, x = "group", y = "value", size = "count",
                       fill = "group", show.legend = FALSE) +
  scale_size(range = c(2, 10)) +
  scale_fill_manual(values = c("A" = "skyblue", "B" = "orange", "C" = "lightgreen")) + #Manual color for better visualization

  geom_text_repel(aes(label = ifelse(threshold_exceeded, paste0(round(value,1),"*"), round(value,1))),
                   size = 3, color = ifelse(data$threshold_exceeded, "red", "black"), box.padding = 0.5, segment.size = 0.2) +

  geom_text_repel(aes(label = paste0("n=", count)), size = 3, color = "black", nudge_y = -2, box.padding = 0.5, segment.size = 0.2) +
  labs(title = "Balloon Plot with Threshold Highlighting") #Added a title for clarity

print(plot)
```

This example adds a `threshold_exceeded` column to the data and uses conditional logic within `geom_text_repel` to change label appearance (color and asterisk) for values exceeding the threshold.  This enhances the visual communication of key data features.


**3. Resource Recommendations:**

*  The `ggplot2` documentation.  Thoroughly understanding its grammar is crucial for creating custom plots and extending functionalities like those shown above.
*  A comprehensive guide on data visualization principles.  Effective data visualization requires a solid understanding of how to present information clearly and avoid misleading interpretations.
*  Documentation for the `ggrepel` package.  This package is invaluable for handling text label overlap in complex plots.  Understanding its parameters is key to optimizing label placement.
