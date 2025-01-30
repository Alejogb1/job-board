---
title: "How can I create a mosaic plot in Python?"
date: "2025-01-30"
id: "how-can-i-create-a-mosaic-plot-in"
---
Mosaic plots, also known as Marimekko charts, are invaluable for visualizing the relationship between categorical variables, particularly when dealing with proportions within subgroups.  My experience working on data visualization projects for a large-scale agricultural analysis firm highlighted their effectiveness in revealing nuanced interactions otherwise obscured by simpler bar charts or pie charts.  Their strength lies in their ability to simultaneously represent the marginal and conditional distributions of multiple categorical variables in a single, easily interpretable visual.  The area of each rectangle in the mosaic plot is directly proportional to the frequency of the corresponding combination of categories, providing an intuitive understanding of the data's structure.

Creating these plots in Python requires a dedicated library as they aren't a standard feature within Matplotlib or Seaborn.  I've primarily utilized the `statsmodels` library for this purpose, finding its functionality robust and its integration with other statistical analyses seamless.  While other libraries might offer similar capabilities, `statsmodels` provided the most straightforward and efficient workflow within my projects, particularly considering the need for statistical tests often coupled with mosaic plot visualizations.

**1. Clear Explanation:**

The core principle behind a mosaic plot's construction lies in the recursive partitioning of a rectangular space.  The initial rectangle represents the total sample size.  This rectangle is then divided proportionally based on the marginal distribution of the first categorical variable. Each resulting sub-rectangle is further divided proportionally based on the conditional distribution of the second categorical variable, given the first.  This process continues for additional categorical variables, resulting in a nested structure where the area of each final rectangle precisely reflects the joint probability of the corresponding categorical combinations.

The key to accurate representation lies in the correct calculation and visualization of these proportions.  `statsmodels` handles this effectively through its `mosaic` function, automatically determining the appropriate partitioning and rendering the plot.  One must carefully consider the order of variables provided to the function as this dictates the hierarchical partitioning and the overall plot's interpretation.


**2. Code Examples with Commentary:**

**Example 1: Basic Mosaic Plot**

```python
import statsmodels.graphics.mosaicplot as mosaic
import pandas as pd

# Sample Data
data = {'Crop': ['Wheat', 'Wheat', 'Corn', 'Corn', 'Soybean', 'Soybean'],
        'Yield': ['High', 'Low', 'High', 'Low', 'High', 'Low'],
        'Region': ['North', 'North', 'South', 'South', 'East', 'East']}
df = pd.DataFrame(data)

# Create Mosaic Plot
mosaic.mosaic(df, ['Crop', 'Yield'], title='Crop Yield by Type')
plt.show()
```

This example demonstrates the simplest application.  The `mosaic` function takes the DataFrame and a list of categorical variables as input.  The order in the list determines the hierarchical partitioning: 'Crop' is the primary division, and 'Yield' is the secondary division within each crop type.  The resulting plot directly shows the proportional relationship between crop type and yield.


**Example 2:  Adding Color and Labels**

```python
import statsmodels.graphics.mosaicplot as mosaic
import matplotlib.pyplot as plt
import pandas as pd

# Sample Data (Same as Example 1)
data = {'Crop': ['Wheat', 'Wheat', 'Corn', 'Corn', 'Soybean', 'Soybean'],
        'Yield': ['High', 'Low', 'High', 'Low', 'High', 'Low'],
        'Region': ['North', 'North', 'South', 'South', 'East', 'East']}
df = pd.DataFrame(data)

# Create Mosaic Plot with Customization
mosaic.mosaic(df, ['Crop', 'Yield'], title='Crop Yield by Type',
              properties={'color': ['lightblue', 'lightcoral']},
              labelizer=lambda k: f'{k[0]} - {k[1]}')
plt.show()
```

Here, we enhance the visual appeal and clarity.  The `properties` argument allows specifying colors for different categories, enhancing visual distinction. The `labelizer` function customizes the labels, providing more descriptive information within each rectangle, improving readability. The `lambda` function elegantly constructs labels by concatenating the categories from the two variables.

**Example 3:  Handling Larger Datasets and Statistical Significance**

```python
import statsmodels.graphics.mosaicplot as mosaic
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulate Larger Dataset
np.random.seed(42)
n = 1000
data = {'Treatment': np.random.choice(['A', 'B', 'C'], size=n),
        'Response': np.random.choice(['Positive', 'Negative'], size=n),
        'Group': np.random.choice(['X', 'Y'], size=n)}
df = pd.DataFrame(data)

# Perform Chi-squared Test (optional, for statistical analysis)
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(df['Treatment'], df['Response'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared test p-value: {p}")

# Create Mosaic Plot
mosaic.mosaic(df, ['Treatment', 'Response'], title='Treatment Response',
              axes_label=False, gap=0.01)  # Removed axes labels for clarity in large datasets
plt.show()
```

This example showcases handling larger datasets, which might necessitate adjustments to presentation for readability. The example also includes a chi-squared test of independence, a common statistical analysis performed alongside mosaic plots to assess the significance of the observed relationships between categorical variables.  This highlights the integration of `statsmodels` with other statistical tools. The `axes_label` parameter is set to `False` for cleaner aesthetics in a larger plot.  The `gap` parameter is decreased to minimize whitespace, improving visual density.


**3. Resource Recommendations:**

1.  The `statsmodels` documentation:  Thoroughly covers the `mosaic` function and its parameters, along with examples and explanations.

2.  A comprehensive data visualization textbook:  Provides a broader context for choosing appropriate visualization techniques and understanding their limitations.  Specific chapters on categorical data visualization would be particularly helpful.

3.  Advanced statistical analysis textbooks focusing on categorical data analysis:  Covers statistical tests like chi-squared tests, which are frequently used in conjunction with mosaic plots.  Understanding the underlying statistical principles strengthens the interpretation of the plots.


By utilizing `statsmodels`, and understanding the principles of proportional partitioning, one can effectively create and interpret mosaic plots for insightful data exploration and communication.  Remember to always consider the context of your data and choose visualization techniques that best highlight the relevant relationships.  Furthermore, coupling visual representations with appropriate statistical testing provides a more robust and reliable analysis.
