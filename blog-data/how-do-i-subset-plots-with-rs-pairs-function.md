---
title: "How do I subset plots with R's `pairs` function?"
date: "2024-12-16"
id: "how-do-i-subset-plots-with-rs-pairs-function"
---

Alright,  Subsetting plots created with R's `pairs` function isn't always immediately obvious, and I’ve definitely spent more time than I care to recall figuring out the exact combinations of arguments to get what I needed back in my early days building statistical models for ecological analysis. The `pairs` function itself is incredibly powerful for visualizing relationships between multiple variables, but sometimes you only need to focus on specific pairings, not the whole shebang. Here's how I’ve approached this over the years, along with some code examples to solidify the techniques.

The core issue stems from the fact that `pairs` is designed to produce a matrix of scatterplots, using combinations of variables across your entire dataset if you provide it with a matrix or data frame. You don't have direct arguments within `pairs` itself to explicitly select which variable combinations to display; instead, you have to be a bit clever in how you supply the data and modify the call.

First, let's talk about a common scenario: when you want to control the specific columns that are included in the pairwise plot. Instead of supplying the whole dataframe, subset it in your function call. Imagine I'm working with a simulated dataset on plant growth where I have measurements for height, stem diameter, leaf area, and flower count. However, I'm primarily interested in the relationship between height and leaf area and diameter and flower count, and not other combinations. I wouldn't necessarily want the full matrix. Here's how I'd do that:

```r
# Simulate a dataframe with plant measurements
set.seed(123)
plant_data <- data.frame(
  height = rnorm(100, 50, 10),
  diameter = rnorm(100, 15, 3),
  leaf_area = rnorm(100, 200, 50),
  flower_count = rpois(100, 10)
)

# Create a subsetted matrix plot
pairs(plant_data[, c("height", "leaf_area", "diameter", "flower_count")], 
      labels = c("Height", "Leaf Area", "Diameter", "Flower Count"), 
      main = "Subset Pair Plot 1: Specific Columns",
      pch = 19, 
      col = "royalblue")

```

In this first example, notice I specifically chose which columns to include when calling `pairs` using `plant_data[, c("height", "leaf_area", "diameter", "flower_count")]`. This is a powerful, direct way to pick out the precise variables you are interested in. The `labels` argument lets me rename columns on the axes, and I've added some cosmetic options using `pch` for plotting symbols and `col` for color. If I only needed height vs leaf area and diameter vs flower count, I'd supply `plant_data[, c("height","leaf_area","diameter","flower_count")]` but manipulate the data to only show what I need, such as a modified data frame, a topic we will cover next.

Now, let's say that the underlying dataset is large and contains a lot of potentially uninteresting variable pairings, and the approach of just selecting columns isn’t granular enough. Here’s where things get a little more interesting; we have to modify the data passed into the `pairs` function by manipulating the data itself, creating a modified dataframe to plot what we need. For instance, I encountered a case in network analysis of city transportation where many variables (population density, road network density, number of public transport stops, etc.) were available, but certain relationships between specific variables like population density vs public transport stops, and road density vs. number of public transport stops were crucial for a particular analysis, but the rest were irrelevant for that analysis. This requires constructing a new dataframe that *only* contains those combinations. Below is the code to demonstrate this:

```r
# Simulate data with many variables (similar to my transport network problem)
set.seed(456)
transport_data <- data.frame(
  population_density = rnorm(100, 1000, 300),
  road_network_density = rnorm(100, 50, 10),
  public_transport_stops = rpois(100, 15),
  bus_stops = rpois(100, 10),
  bike_lanes = rnorm(100, 20, 5),
  car_ownership = rnorm(100,0.6,0.2)
)

# Create a custom dataframe with only the combinations of interest
custom_transport_data <- data.frame(
  x1 = transport_data$population_density,
  y1 = transport_data$public_transport_stops,
  x2 = transport_data$road_network_density,
  y2 = transport_data$public_transport_stops
)

# Plot the selected variable combinations using pairs
pairs(custom_transport_data, 
      labels = c("Pop. Density", "Pub. Transport Stops", "Road Density", "Pub. Transport Stops"),
      main = "Subset Pair Plot 2: Custom Dataframe",
      pch = 17, 
      col = "darkgreen")
```

Here, I've built a new data frame `custom_transport_data` where I’ve explicitly defined the pairs of variables I want to explore, renaming them `x1`, `y1`, `x2`, and `y2`. When I call the `pairs` function now, it renders the desired scatterplots without the additional extraneous combinations. Notice how I include the same variable, number of `public_transport_stops` as both y1 and y2 because I’m curious about that variable’s relationship with two other variables.

There is one other way, perhaps more useful if you have many variables but only a few specific combinations, involving using logical indexing on the columns. This often simplifies my process. Imagine you have ten or fifteen or even more columns in your dataframe, and you have very specific pairings that aren’t necessarily in a simple sequence. This allows for greater flexibility. In a genetics study, for instance, we needed to check the correlation between specific genes on different chromosomes where the columns were not arranged in an orderly manner. Here's a way to select particular columns using their names, and we can utilize that in the `pairs` call:

```r
# Simulate a genetic dataset with numerous gene expression values
set.seed(789)
gene_data <- data.frame(
  gene_a = rnorm(100, 5, 1),
  gene_b = rnorm(100, 10, 2),
  gene_c = rnorm(100, 8, 1.5),
  gene_d = rnorm(100, 12, 2.5),
  gene_e = rnorm(100, 6, 1),
  gene_f = rnorm(100, 11, 2.1),
  gene_g = rnorm(100, 7, 1.6)
)

# Select the specific columns of interest using column names
columns_of_interest <- names(gene_data) %in% c("gene_a", "gene_c", "gene_f")
# Plot the selected variable combinations using pairs
pairs(gene_data[, columns_of_interest],
      labels = c("Gene A", "Gene C", "Gene F"),
      main = "Subset Pair Plot 3: Logical Indexing",
      pch = 15, 
      col = "purple")
```
Here, I use the `%in%` operator to select only specific column names that are of interest, then use the resulting logical vector `columns_of_interest` to index the dataframe in the `pairs` call. This approach is both concise and easily modified to include different combinations of variables.

To further enhance your understanding, I highly recommend diving into “ggplot2: Elegant Graphics for Data Analysis” by Hadley Wickham; while not directly focusing on `pairs`, it offers a deeper understanding of data visualization principles, which can complement the usage of base R plotting. Also, exploring the various base plotting packages in the R manual is helpful for understanding how different arguments are used. Additionally, you might find “The R Graphics Book” by Paul Murrell invaluable for comprehensive coverage of the R graphics system. In summary, while `pairs` doesn't offer direct subsetting arguments, clever data manipulation and column selection allow you to produce exactly the plots you need for your analysis.
