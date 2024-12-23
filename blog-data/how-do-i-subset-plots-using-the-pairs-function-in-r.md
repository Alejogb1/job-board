---
title: "How do I subset plots using the pairs function in R?"
date: "2024-12-23"
id: "how-do-i-subset-plots-using-the-pairs-function-in-r"
---

Let's dive right into the particulars of subsetting plots when employing the `pairs` function in R. I've certainly spent a fair bit of time navigating this myself, especially when working with larger datasets, and I'm happy to share some insights that have proven useful over the years. It's easy to get a bit overwhelmed when dealing with a scatterplot matrix displaying everything and the kitchen sink; subsetting is absolutely crucial for a clear, focused visualization.

The `pairs` function in R is fantastic for quickly generating scatterplot matrices, but it doesn't directly offer a 'subset' argument in the way you might expect with, say, a data frame. This is a common point of confusion, particularly if you’re accustomed to subsetting in other R contexts. Instead, subsetting plots with `pairs` requires a slightly different approach, primarily involving manipulating the data before it is passed to the function or using custom panel functions to control what is displayed.

Generally, there are three main techniques that I’ve found effective, and I’ll explain each in detail, including the code and why each approach works well, along with their individual pros and cons.

First, and perhaps the most straightforward approach, is to subset your data frame before using `pairs`. This is conceptually simple: you’re pre-selecting the rows or columns that you’re interested in and then visualizing only this curated dataset. This method excels in terms of readability and ease of understanding, and I often use it as my first choice due to its simplicity.

```R
# Example 1: Subsetting by rows
data("iris")
subset_iris <- iris[iris$Species == "setosa", ]
pairs(subset_iris[, 1:4], main="Pairs plot of Setosa Iris")
```
In this example, the iris dataset (a standard, built-in R dataset) is used. Here, `iris[iris$Species == "setosa", ]` creates a new data frame that only includes the rows where the `Species` column is equal to "setosa." By using the comma followed by the bracket indexing, I'm selecting every column (1:4) of this new subset. Thus, the resulting pairs plot exclusively displays scatter plots for only the setosa species. This is an effective way to target a section of the data based on a column value in our data. The main advantage? Clarity and directness - you’re not altering how the plot operates but rather what it operates *on*.

Another frequently encountered scenario requires subsetting by columns. This comes into play often when dealing with high-dimensional datasets where only a selected subset of variables is relevant for a particular visualization.

```R
# Example 2: Subsetting by columns
data("iris")
pairs(iris[, c("Sepal.Length", "Petal.Length", "Petal.Width")], main="Pairs plot of selected iris variables")
```

This snippet illustrates subsetting by columns via the bracket indexing technique again. Here, we are directly selecting three columns of the iris dataset, namely: `"Sepal.Length"`, `"Petal.Length"`, and `"Petal.Width"` and feed only these to `pairs`. This approach is very handy when working with datasets containing many variables and you want to focus only on the variables of interest without having to manually redefine new data frames. This kind of column selection is extremely common and straightforward, avoiding the creation of temporary data frames, but does change how you're seeing the overall dataset.

Finally, a third, and more advanced method, is to use the `panel` argument within `pairs` to generate customized plots within individual subpanels of the pairs plot matrix. It offers more flexibility but requires a bit more understanding of R's plotting internals. This lets you add conditional visualizations within your plots and I have used it quite a bit when looking for patterns within a complex dataset.

```R
# Example 3: Custom panels
data("iris")

custom_panel <- function(x, y, ...){
    points(x, y, pch = 20, col = "black")
     if (cor(x, y) > 0.7){
       abline(lm(y ~ x), col = "blue")
       text(mean(range(x)), mean(range(y)),
             labels = paste0("corr=", round(cor(x, y), 2)), cex = 0.7, col="red")
    }
}

pairs(iris[, 1:4], panel=custom_panel, main = "Pairs plot with correlation > 0.7")
```
In this example, we define a custom `custom_panel` function. This function is called for each subpanel in the pairs plot, getting access to x and y values of the sub-panel data. We first plots the points for a specific set of xy variable in the particular sub-panel. Then, this function calculates the correlation and adds a regression line and annotates the correlation if it’s greater than 0.7 in each panel. This demonstrates that you aren’t simply restricted to scatter plots, and you can create conditional plots and annotations directly. This allows for very complex visualizations within a pairs plot and provides much greater granularity, but also increases the complexity of your code.

Which approach is 'best' really depends on the specific situation. Simple row or column subsetting, like in the first two examples, is often sufficient for most common tasks. However, when you need more fine-grained control or want to conditionally visualize features, using a custom panel function like in the third example becomes essential.

For those looking to further improve their skills and understanding of R’s visualization capabilities, I suggest delving deeper into the following resources:

*   **"R Graphics" by Paul Murrell:** A comprehensive guide to R's graphics system, covering both base graphics and the `grid` package, which underlies many complex plotting libraries.
*   **"ggplot2: Elegant Graphics for Data Analysis" by Hadley Wickham:** While ggplot2 is not directly used with the pairs function here, it provides a very useful approach for creating complex and highly customizable plots, learning this framework will give you greater control over all forms of data visualisation.
*  **"Advanced R" by Hadley Wickham:** Provides a detailed technical explanation of R's internal workings and programming paradigm which will also further improve your understanding of complex custom panel functions and generally will improve your ability to write elegant and efficient code.

These books are readily available and are considered standard texts for serious R programmers. Mastering these resources will significantly enhance your abilities in data visualization and analysis in R, moving beyond the basics to more sophisticated techniques.

The main takeaway, in summary, is that while `pairs` doesn't directly include subset arguments, you have powerful options through pre-subsetting your dataset or by using custom panel functions. It's a versatile function when coupled with the right techniques, allowing you to create effective visualizations from even complex, high-dimensional data. Keep practicing and exploring, and you'll quickly master these methods for your own data visualization work.
