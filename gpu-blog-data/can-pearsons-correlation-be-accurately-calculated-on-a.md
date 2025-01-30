---
title: "Can Pearson's correlation be accurately calculated on a linear-log scatterplot?"
date: "2025-01-30"
id: "can-pearsons-correlation-be-accurately-calculated-on-a"
---
Pearson's correlation coefficient, a measure of linear association, is fundamentally predicated on a linear relationship between two variables.  Applying it directly to data visualized on a linear-log scatterplot requires careful consideration, as the logarithmic transformation fundamentally alters the data's distribution and the nature of the relationship.  My experience working on financial time series analysis, specifically modeling asset price volatility, has highlighted this limitation repeatedly.  While seemingly straightforward, the accuracy of the result depends heavily on the underlying data and the goals of the analysis.

**1. Clear Explanation:**

Pearson's correlation (r) quantifies the strength and direction of a *linear* relationship.  It's defined as the covariance of two variables divided by the product of their standard deviations.  Crucially, this calculation assumes that the relationship between the variables is best approximated by a straight line. A linear-log scatterplot, where one axis (typically the y-axis) is logarithmically scaled, suggests a non-linear relationship;  specifically, it implies that a change in the x-variable leads to a proportional change in the *logarithm* of the y-variable.  Thus, a direct application of Pearson's correlation to the raw data presented in a linear-log plot will yield a value that may not accurately reflect the true strength of the underlying relationship.  The correlation will be artificially deflated if the data follow a power-law relationship, and may be misleading if other non-linear relationships are present.

The appropriate approach depends on the research question. If the goal is to assess the linear correlation between the x-variable and the logarithm of the y-variable, then calculating Pearson's correlation directly on the transformed data is valid.  This is often appropriate when the underlying relationship is suspected to be exponential or power-law in nature.  However, if the goal is to quantify the correlation between the raw variables (ignoring any transformations), then other correlation measures more suitable for non-linear relationships, such as Spearman's rank correlation or Kendall's tau, should be considered.  Further, techniques like fitting a curve (exponential, power, etc.) and then using correlation on the fitted values should be explored.

**2. Code Examples with Commentary:**

The following examples use Python with the `numpy` and `scipy` libraries.  Assume `x` and `y` represent the original data.

**Example 1: Direct Application (Potentially Inaccurate):**

```python
import numpy as np
from scipy.stats import pearsonr

# Raw data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 40, 80, 160])

#Direct Pearson's correlation on raw data (linear-log relationship)
correlation, p_value = pearsonr(x, y)

print(f"Pearson's correlation (raw data): {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
```

This example demonstrates a naive application of Pearson's correlation. Because the underlying relationship is exponential, the correlation coefficient will not fully capture the strong relationship between x and y.  The low correlation is a consequence of the methodology, not the lack of relationship.


**Example 2: Correlation on Log-Transformed Data:**

```python
import numpy as np
from scipy.stats import pearsonr
import numpy as np

# Raw data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 40, 80, 160])

# Log-transform y-data
log_y = np.log(y)

# Pearson's correlation on x and log(y)
correlation, p_value = pearsonr(x, log_y)

print(f"Pearson's correlation (log-transformed y): {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
```

This approach is appropriate if interest lies in the relationship between x and log(y). A high correlation coefficient here suggests a strong exponential relationship between x and y.


**Example 3: Non-parametric Correlation:**

```python
import numpy as np
from scipy.stats import spearmanr

# Raw data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 40, 80, 160])

# Spearman's rank correlation
correlation, p_value = spearmanr(x, y)

print(f"Spearman's rank correlation: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
```

This utilizes Spearman's rank correlation, a non-parametric method robust to non-linear relationships and outliers.  It measures the monotonic relationship between x and y, offering an alternative assessment when a linear correlation is questionable.  The choice between Spearman's and Kendall's tau depends on the specific needs and robustness requirements of the analysis.


**3. Resource Recommendations:**

For a deeper understanding of correlation and its various forms, I recommend consulting standard statistical textbooks.  Focus on chapters covering correlation analysis, hypothesis testing, and non-parametric methods.  The documentation for statistical software packages like R and Python's `scipy` library are also invaluable resources, providing detailed explanations and usage examples of various correlation functions. Finally, reviewing research papers within your specific field will help you determine the accepted practices for dealing with non-linear relationships and the most appropriate correlation measures for the problem at hand.  A solid grasp of regression analysis will also enhance your ability to model and interpret non-linear relationships.
