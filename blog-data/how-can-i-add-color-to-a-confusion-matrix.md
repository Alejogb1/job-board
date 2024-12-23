---
title: "How can I add color to a confusion matrix?"
date: "2024-12-23"
id: "how-can-i-add-color-to-a-confusion-matrix"
---

Alright, let’s talk about adding some color to confusion matrices. It’s a task I’ve tackled more times than I care to count, often finding myself needing to communicate model performance to stakeholders who might not be as steeped in the intricacies of machine learning metrics. A grayscale matrix, while perfectly functional, sometimes just doesn’t cut it when it comes to clarity and impact. My own baptism by fire came years back, on a project involving a rather finicky spam classifier—the need for visually distinct performance characteristics was paramount then. So, let's dive into how we can accomplish this effectively, aiming for readability and precision.

The core of this process involves mapping numerical values in your confusion matrix to color intensities. This isn’t just about aesthetic appeal; it’s about guiding the viewer’s eye to critical regions of the matrix. Generally, higher values, representing larger counts of true positives, true negatives, false positives, or false negatives, are associated with stronger, more saturated colors, and lower values with paler shades or even transparency. The choice of color map (or colormap) is crucial, and a carefully chosen one can make all the difference between a quickly understood graphic and a frustrating, perplexing display.

Let's start with the fundamentals. A confusion matrix, at its core, is a 2D array (or a matrix, if you prefer the mathematical term), where each cell represents a combination of predicted and actual class labels. For instance, in a binary classification problem, you might have entries for true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). Extending this to multiclass problems just means you'll have more rows and columns, each corresponding to a particular class label.

Now, how do we color-code these values? We use colormaps. These are typically predefined mappings of a scalar value (the numerical entry in our matrix) to a color, often represented as RGB (Red, Green, Blue) triplets or RGBA (Red, Green, Blue, Alpha) quadruplets. Libraries like `matplotlib` in Python or similar libraries in other languages provide an extensive palette of these colormaps, ranging from sequential (where intensity increases monotonically) to diverging (where a central neutral value is mapped to a neutral color, and extremities diverge to other hues).

Here’s where my experience has been beneficial: not all colormaps are equally effective. Sequential colormaps like `viridis` or `magma` tend to work well when you're just interested in the relative magnitude of values, allowing you to easily see where values are higher or lower. Diverging colormaps, such as `coolwarm` or `RdBu`, are more useful when you want to emphasize deviations from a central, neutral value – which often isn't typically the case for confusion matrices, but might be useful if analyzing the difference in misclassification. For pure confusion matrix display I’ve found sequential color maps generally the better choice.

Let’s put this into action with a few code examples, using Python and `matplotlib` as my go-to tools:

**Example 1: Basic Heatmap of a Binary Confusion Matrix**

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Example confusion matrix
cm = np.array([[75, 25],
               [10, 90]])

# Create a heatmap using a sequential colormap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('Binary Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

This snippet uses the `seaborn` library, which builds upon `matplotlib`, to generate the heatmap. The `annot=True` argument displays the numerical values in each cell, and `fmt="d"` specifies that they should be formatted as integers. `cmap="Blues"` sets the colormap to a sequential blue scale. A colorbar is also included, which is invaluable for understanding the mapping from numerical values to colors.

**Example 2: Multiclass Confusion Matrix with Custom Labels**

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Example multiclass confusion matrix
cm = np.array([[50,  5,  2],
              [ 3, 60,  7],
              [ 1,  4, 70]])

class_names = ["Class A", "Class B", "Class C"]

# Creating the heatmap with custom labels
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('Multiclass Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

This example shows the process for a multiclass matrix, and illustrates how to replace default numeric labels with descriptive class labels using `xticklabels` and `yticklabels` arguments. It uses the sequential colormap `viridis` which helps to see the distribution.

**Example 3: Adding Normalization for Percentage Representation**

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Example confusion matrix
cm = np.array([[75, 25],
               [10, 90]])

# Normalize the confusion matrix for percentage representation
cm_normalized = cm / np.sum(cm, axis=1, keepdims=True)

# Create a heatmap with normalized values
plt.figure(figsize=(8,6))
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Greens", cbar=True)
plt.title('Normalized Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

Here we normalize the matrix, dividing each row by the sum of that row. This converts the raw counts to percentages within each true class. The formatting has also been changed to `.2%`, to display values as percentages with two decimal points. This can be extremely useful if you want to analyze per-class performance ratios irrespective of class size. `Greens` is used as colormap for emphasis.

Beyond these code examples, there are a few practices that I’ve found invaluable over the years: Always, *always* include a colorbar (the `cbar=True` argument in these examples). This is the key to allowing the viewer to correlate the color intensities with numerical values. Without it, the visual interpretation is subjective and often misleading. Another critical aspect is readability. Ensure sufficient contrast between colors, and if you are placing the numerical values inside each cell, ensure that the font color has enough contrast to be easily visible against the color of the cell. A white font on a pale background, for example, is a recipe for an unreadable visualization.

In terms of resources, I would highly recommend looking into the documentation of `matplotlib` and `seaborn` in Python. For a deeper understanding of colormaps, you can also check out the research paper *“Rainbow Color Map (Still) Considered Harmful”* by Kenneth Moreland. There is also an abundance of information in Edward Tufte’s books, particularly *“The Visual Display of Quantitative Information,”* which, while not specifically about confusion matrices, provides foundational knowledge on creating effective visualizations. For the mathematical aspects of matrices, Gilbert Strang's *“Linear Algebra and Its Applications”* is a fantastic resource to improve your grasp.

In closing, adding color to a confusion matrix is more than just a cosmetic enhancement. It's about improving clarity, enhancing the interpretability, and facilitating communication of critical model performance metrics. The examples provided, combined with the suggested practices and resources, should give you a solid foundation for creating insightful and impactful visualizations. Remember, the goal is to present complex data in a way that's not only visually appealing but also informative and easy to grasp.
