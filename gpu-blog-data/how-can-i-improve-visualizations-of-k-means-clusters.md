---
title: "How can I improve visualizations of K-means clusters using fviz_clusters()?"
date: "2025-01-30"
id: "how-can-i-improve-visualizations-of-k-means-clusters"
---
The default output of `fviz_clusters()` in R, while functional for initial exploration, often lacks the clarity and nuanced presentation required for formal analysis or reporting. I've frequently encountered situations where the automatic scaling and default parameter choices obscured meaningful patterns within my K-means results, particularly when dealing with higher-dimensional datasets.

My experience in analyzing gene expression data using K-means highlighted this issue early on. The standard scatter plots produced by `fviz_clusters()`, relying solely on the first two principal components, were insufficient to adequately represent cluster separations when dimensionality was high. This often resulted in overlapping clusters and misinterpretations of the cluster membership. The issue wasn't with the algorithm itself, but rather with the visual depiction of its output. I then learned to use the function's customization capabilities for better data representation.

The `fviz_clusters()` function, part of the `factoextra` package, primarily produces visualizations of K-means results using a dimensionality reduction technique (typically Principal Component Analysis, PCA) when the data exceeds two dimensions. It is important to understand that the resulting plot displays clusters in the reduced space, which can distort the underlying relationships. Therefore, improving visualizations with this function requires careful consideration of data projection, plot aesthetic, and information richness. The function returns a ggplot2 object allowing granular customizations.

Here's how I’ve approached enhancing K-means visualizations using `fviz_clusters()`:

**1. Adjusting the Dimensionality Reduction and Principal Component Selection:**

The most significant improvement stems from controlling the dimensionality reduction process itself. `fviz_clusters()` by default uses PCA to reduce the dataset to two dimensions for visualization. However, the first two principal components do not always capture the most variance contributing to cluster separation. To address this, I utilize the `pc.biplot` argument. This allows me to plot the clusters in terms of their original dimensions. This is particularly useful when the dimensions have interpretable meaning. When using PCA, I examine the scree plot using `fviz_pca_var()` to determine the most relevant components to plot. Selecting relevant PC pairs instead of simply defaulting to PC1 and PC2 often provides a much clearer separation.

**2. Enhancing Plot Aesthetics for Clarity:**

A visually cluttered plot hinders proper interpretation. Therefore, I use the ggplot2 capabilities offered by `fviz_clusters()` to refine visual aspects. This includes specifying colors using the `palette` argument for improved cluster differentiation, adjusting marker sizes to avoid overplotting, modifying the legend positioning, and adding informative titles/labels. Furthermore, I use the `ellipse.type` argument to display cluster boundaries, and have found that `'convex'` provides good cluster delimitation without being visually aggressive compared to `'confidence'` ellipses. Additionally, controlling point transparency (`alpha` argument in geom_point) is important to manage overplotting when there are dense clusters.

**3. Incorporating Additional Information:**

I often augment the base plot to provide richer information. For instance, I overlay cluster centroids using `geom_point()` with contrasting colors and sizes for better visual identification. If the dataset includes auxiliary data about the points, such as a class label, I use these to highlight subgroups within the cluster by encoding the information using fill or shape aesthetics. When examining high-dimensional data, it becomes useful to plot the cluster centroids in a parallel coordinate plot to understand the features associated with each cluster. This goes beyond simple scatterplots for the reduced data.

Here are some practical examples with commentary:

**Example 1: Customizing Colors and Point Size**

```R
library(factoextra)
library(cluster)

# Generate synthetic data with three clusters
set.seed(123)
data <- rbind(matrix(rnorm(50, mean = 0, sd = 1), ncol = 2),
              matrix(rnorm(50, mean = 5, sd = 1), ncol = 2),
              matrix(rnorm(50, mean = 10, sd = 1), ncol = 2))
colnames(data) <- c("Feature 1", "Feature 2")


# Perform K-means clustering
kmeans_result <- kmeans(data, centers = 3, nstart = 25)

# Basic plot (unmodified)
print(fviz_cluster(kmeans_result, data = data,
                    main = "Default K-means Plot"))


# Customized plot with specified colors and point size
print(fviz_cluster(kmeans_result, data = data,
                    palette = c("blue", "green", "red"),
                    geom = "point",
                    pointsize = 3,
                    main = "Customized K-means Plot"))
```

In this example, the initial `fviz_cluster()` output uses the default color scheme. The second call utilizes the `palette` argument to specify custom colors for each cluster, improving differentiation. I have also used `pointsize = 3` to make the points more prominent and visually accessible.

**Example 2: Using Original Dimensions for Plotting and Adding Centroids**

```R
library(factoextra)
library(cluster)
library(ggplot2)

# Generate synthetic data with three clusters, now 3-Dimensional
set.seed(123)
data <- rbind(matrix(rnorm(75, mean = 0, sd = 1), ncol = 3),
              matrix(rnorm(75, mean = 5, sd = 1), ncol = 3),
              matrix(rnorm(75, mean = 10, sd = 1), ncol = 3))
colnames(data) <- c("Feature 1", "Feature 2", "Feature 3")

# Perform K-means clustering
kmeans_result <- kmeans(data, centers = 3, nstart = 25)


# Plot using original dimensions with centroids, using pc.biplot=FALSE
plot_obj <- fviz_cluster(kmeans_result, data = data,
                         ellipse.type = "convex",
                         axes = c(1,2),
                         pc.biplot = FALSE,
                         main = "K-means Clusters in Original Dimensions",
                         xlab = "Feature 1",
                         ylab = "Feature 2",
                         legend = "right" ) +
                         geom_point(data = as.data.frame(kmeans_result$centers),
                                     aes(x= V1, y= V2),
                                     color="black",
                                     size = 5,
                                     shape = 8)


print(plot_obj)
```
This example shifts from PCA-based reduction to using the original dimensions for plotting via `pc.biplot = FALSE` and selects axes `c(1,2)` to display dimensions one and two. It also adds the cluster centroids as diamond shapes using `geom_point`. This allows for analysis in the initial dimension space.

**Example 3: Using PCA, Selecting PCs, and Using Convex Hulls**
```R
library(factoextra)
library(cluster)
library(ggplot2)

# Generate synthetic data with three clusters, now 5-Dimensional
set.seed(123)
data <- rbind(matrix(rnorm(125, mean = 0, sd = 1), ncol = 5),
              matrix(rnorm(125, mean = 5, sd = 1), ncol = 5),
              matrix(rnorm(125, mean = 10, sd = 1), ncol = 5))
colnames(data) <- paste("Feature",1:5)

# Perform K-means clustering
kmeans_result <- kmeans(data, centers = 3, nstart = 25)

# Perform PCA
pca_result = prcomp(data, center = TRUE, scale. = TRUE)

# Scree plot for PC selection
print(fviz_pca_var(pca_result,
                    title = "PCA Scree Plot"))

# Using PCA but selected components.
plot_obj = fviz_cluster(kmeans_result, data = data,
                     ellipse.type = "convex",
                     axes = c(2,3),
                     main = "K-means Clusters in PCA Space",
                     legend = "right" )

print(plot_obj)
```

This example first computes PCA and displays the scree plot. Then, using the information in the scree plot, this example plots clusters in PCA space based on components 2 and 3. It uses convex hulls to identify cluster boundaries and makes use of the default color pallet.

**Resource Recommendations:**

To deepen your understanding, I recommend exploring the documentation of the `factoextra` and `ggplot2` packages. These will provide extensive insights into parameter control and more advanced plotting techniques. The book "R for Data Science" offers invaluable context on data visualization principles using ggplot2. Also, consulting statistical texts and publications that focus on clustering and dimensionality reduction is key for a more conceptual understanding and appropriate implementation choices. These resources should offer the required material to further improve cluster visualizations in a wide variety of scenarios.
