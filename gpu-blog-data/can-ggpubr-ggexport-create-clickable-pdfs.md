---
title: "Can ggpubr (ggexport) create clickable PDFs?"
date: "2025-01-30"
id: "can-ggpubr-ggexport-create-clickable-pdfs"
---
The `ggpubr` package, specifically its `ggexport` function, does not inherently generate clickable PDFs in the sense of interactive elements like hyperlinks or form fields directly embedded within the plots themselves. Instead, `ggexport` is designed to efficiently save static plots and arrange them into presentation-ready formats, including PDF, but these output PDFs are not interactive. My experience in developing data visualization pipelines for biological research projects has repeatedly underscored this limitation when we've needed interactive components within our reports.

The primary function of `ggexport` is to take one or more `ggplot2` plot objects and render them into a desired output file format. It leverages underlying graphics devices within R, such as `pdf()`, to accomplish this. While the generated PDFs can be navigated by page, and text annotations within plots are selectable, the visual elements of the plot—lines, shapes, bars, points—are rendered as static images embedded within the PDF. These images cannot be directly interacted with beyond standard PDF reader functions like zooming and panning. The lack of direct interactivity stems from the fundamental way `ggplot2` and its related packages operate. They primarily produce static vector graphics (or rasterized versions), not interactive web-based graphics like those produced by libraries such as `plotly` or `shiny`. The rendered PDF output retains this static nature. `ggexport` provides valuable layout options, such as arranging plots into multi-panel figures or creating PDF slideshows, but these features are orthogonal to the creation of interactive plot elements.

Here are three examples illustrating different uses of `ggexport` and the limitations related to interactivity:

**Example 1: Basic PDF Export**

```R
library(ggplot2)
library(ggpubr)

# Sample data
data <- data.frame(x = 1:10, y = rnorm(10))

# Create a ggplot
plot1 <- ggplot(data, aes(x, y)) +
  geom_line() +
  ggtitle("Simple Line Plot")

# Export to PDF
ggexport(plot1, filename = "simple_plot.pdf")

```

This code demonstrates the most basic usage of `ggexport`. A simple line plot is created using `ggplot2`, and then `ggexport` saves this plot as a `simple_plot.pdf` file. The generated PDF will display the line plot, with the axis labels and title as text, but the line representing the data will be static. There is no method of clicking or hovering to reveal underlying data points or related information, even if the underlying text within the plot was clickable (which it is, in the sense that you can select it with your cursor). The plot elements are rendered as part of a static visual graphic. The interactivity one might expect in a web application is simply absent.

**Example 2: Multi-plot PDF Arrangement**

```R
library(ggplot2)
library(ggpubr)

# Sample Data
data2 <- data.frame(group = rep(c("A", "B"), each = 10), val = rnorm(20))

# Create two plots
plot2 <- ggplot(data2, aes(group, val)) +
  geom_boxplot() + ggtitle("Box Plot")
plot3 <- ggplot(data2, aes(val)) +
  geom_histogram() + ggtitle("Histogram")

# Arrange the plots in a grid and export to PDF
ggexport(plot2, plot3, filename = "multi_plot.pdf",
         nrow = 1, ncol = 2)

```
Here, I have created two different plots: a box plot and a histogram from a slightly different dataset. Using the `nrow` and `ncol` arguments, these two plots are arranged side-by-side within the output `multi_plot.pdf` file. This demonstrates `ggexport`'s capability to arrange multiple plots into a single PDF. Again, while the plot titles and axis labels are selectable text, the box plots and histogram are static images, and do not have any direct interactivity embedded in them. Even with this grid layout, no component of the plots is click-able in the way one might expect interactive charts to be. They remain static.

**Example 3: Export with Custom Dimensions**

```R
library(ggplot2)
library(ggpubr)

#Sample Data
data3 <- data.frame(x=1:20, y = runif(20))

#Create Scatterplot
plot4 <- ggplot(data3, aes(x, y))+
  geom_point()+
  ggtitle("Custom Scatterplot")

# Export with custom width and height
ggexport(plot4, filename = "custom_plot.pdf",
         width = 8, height = 6)

```

This final example shows how to control the dimensions of the exported plot. The `width` and `height` arguments are set to specify the size of the output PDF. While the plot itself is well-rendered and the size is as specified in inches, the fundamental fact that the plot's elements lack any interactive functionality remains. This example reinforces that `ggexport`'s functionality is oriented towards producing high-quality, static visualisations.

In summary, while `ggexport` is excellent at rendering `ggplot2` plots into PDF format and arranging these plots within a report, it is not designed to create interactive elements within the produced PDF documents. This lack of interactivity is a core feature of its static rendering approach. To achieve interactivity in plots, alternative approaches and tools are necessary.

For users who need interactive plots, a few alternatives exist. The `plotly` package can create interactive web-based graphics, often using JavaScript, that can be embedded into HTML reports or deployed as web applications using frameworks such as `shiny`. Similarly, the `rgl` package allows the creation of 3D interactive plots which can also be exported or interacted with through web interfaces. For producing interactive documents which incorporate R plots and other content, consider using `rmarkdown` to generate web pages with `plotly` or other interactive graphic libraries. The core approach involves building and publishing a web-based deliverable, rather than relying on a PDF. Various tutorials and examples are available both for the base packages (`plotly`, `shiny`, `rgl`) as well as for their integration with `rmarkdown` in online documentation and books. These avenues, while increasing development complexity, ultimately allow one to bring interactive features to one’s data visualizations, something that is not available to the PDF output by `ggpubr`.
