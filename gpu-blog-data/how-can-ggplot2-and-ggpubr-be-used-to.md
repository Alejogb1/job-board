---
title: "How can ggplot2 and ggpubr be used to globally change tag size and color?"
date: "2025-01-30"
id: "how-can-ggplot2-and-ggpubr-be-used-to"
---
When working on a large report involving numerous ggplot2 visualizations, I often find myself needing to adjust the appearance of plot tags (e.g., axis labels, titles, legend text) consistently across all figures. The process of modifying these elements on a plot-by-plot basis becomes tedious and error-prone. While theme functions in ggplot2 allow for customization, directly modifying tag sizes and colors globally, without having to repeat the same theme specifications, requires a combination of ggplot2's theme system and careful function application. This avoids the pitfall of hardcoding visual parameters within each plot creation. I will describe this process using several worked examples.

First, understanding how ggplot2's theme elements are structured is critical. A ggplot2 theme is a collection of settings that control the visual appearance of a plot. These settings include aspects like background color, gridlines, text size, and font faces. The specific text elements we are concerned with, such as titles, axes labels, axis text, and legend text, are controlled via dedicated theme elements that allow customization of size, color, and family (font). These elements have names corresponding to the graphical component they modify; for instance, `plot.title` controls the plot title, `axis.text.x` controls text on the x-axis, and `legend.text` controls the legend text. These theme elements are modified using the `theme()` function and specified through the `element_text()` function. This function accepts numerous parameters such as `size` and `color` that define appearance.

To affect these settings globally, we'll utilize functions that encapsulate theme modifications and apply them after plot generation. A simple approach is to create a custom function that returns a modified theme. This function can be applied to every ggplot2 plot we generate using the `+` operator. This avoids redundancy in code. We can take this further by creating functions that handle color adjustment. These changes can then be combined using ggplot2’s composition mechanism.

Here’s the first example demonstrating how to change the size of all tag elements.

```R
library(ggplot2)

# Define a function to globally change tag size
adjust_tag_size <- function(base_size = 12) {
  theme(
    plot.title = element_text(size = base_size + 4),
    axis.title.x = element_text(size = base_size + 2),
    axis.title.y = element_text(size = base_size + 2),
    axis.text.x = element_text(size = base_size),
    axis.text.y = element_text(size = base_size),
    legend.text = element_text(size = base_size)
  )
}


# Create a sample plot
plot1 <- ggplot(mtcars, aes(x = mpg, y = wt)) +
  geom_point() +
  labs(title = "MPG vs. Weight", x = "Miles per Gallon", y = "Weight") +
    theme_minimal()

# Apply the custom theme function
plot1 + adjust_tag_size(14)

```

In this example, the function `adjust_tag_size` creates a `theme` object with modifications to all relevant text elements. `base_size` provides a central adjustment point; the plot and axis titles are slightly larger than the base size. The plot `plot1` demonstrates the typical process; data is defined, aesthetic mapping is set, geometries are added, and labels are added, as a minimal base plot. Applying the `adjust_tag_size(14)` function to it increases the base size to 14 points, adjusting the tag elements accordingly. Each element’s size is defined in terms of the base size, making adjustment simple.

Next, consider globally changing the tag colors. This follows similar functional programming, using `element_text`, to define the required text element.

```R

# Function to globally change tag color
adjust_tag_color <- function(tag_color = "blue"){
  theme(
    plot.title = element_text(color = tag_color),
    axis.title.x = element_text(color = tag_color),
    axis.title.y = element_text(color = tag_color),
    axis.text.x = element_text(color = tag_color),
    axis.text.y = element_text(color = tag_color),
    legend.text = element_text(color = tag_color)
  )
}

# Apply the color adjustment function
plot1 + adjust_tag_size(14) + adjust_tag_color("red")
```

The `adjust_tag_color` function takes `tag_color` as an input; every text element’s color is set to this value. Applying both the size adjustment function and color adjustment function by using the `+` operator creates a plot with both modifications. By combining these two functions, it becomes simple to manage style across many plots in a standardized manner.

Finally, a more complex example involving ggpubr. When using `ggpubr` to create arrangements or composite plots, it's still possible to apply these adjustments. We need to ensure we apply the theme after the plot is created using `ggarrange`.

```R

library(ggpubr)

# Create a second plot
plot2 <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Sepal Length vs. Width", x = "Sepal Length", y = "Sepal Width") +
  theme_minimal()


# Arrange plots using ggarrange
arranged_plot <- ggarrange(plot1, plot2, ncol = 2)


# Apply theme adjustments after arrangement
arranged_plot + adjust_tag_size(10) + adjust_tag_color("darkgreen")
```

Here, `ggarrange` combines `plot1` and `plot2` into a single arranged figure. The theme adjustments are applied after the arrangement is created to the whole composition. This demonstrates that functions can be consistently used after plot construction, either individually, or within arrangements.

In practice, it is important to use a carefully chosen set of theme adjustments, as setting these globally may interfere with particular plot requirements. For example, using a very small `base_size` can make text unreadable on some plots. If specific plots do need different values, adjustments to the default theme can be applied after the global adjustments. This allows for a good balance between standardization and flexibility. Furthermore, applying such changes as early as possible during plot creation is advised, in order to ensure a more robust and predictable workflow.

For further understanding of ggplot2 themes, consult “ggplot2: Elegant Graphics for Data Analysis” by Hadley Wickham, specifically the chapter on themes. In addition, the online documentation for ggplot2 provides a comprehensive reference on `theme()` and its many components. To gain further insights into plot arrangements, the `ggpubr` documentation explains how arrangements are constructed and modified. While this documentation can be specific to the functions available, it is generally a good resource to understand plot modification and compositing techniques. Finally, consider practicing modifying different parameters via the `element_text` function, experimenting to understand how to generate the desired presentation. Using this practice to build specific functions for common stylistic choices, provides a robust template that will streamline reporting workflows, and improve the overall visual consistency of graphical outputs.
