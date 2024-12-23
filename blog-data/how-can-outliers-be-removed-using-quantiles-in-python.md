---
title: "How can outliers be removed using quantiles in Python?"
date: "2024-12-23"
id: "how-can-outliers-be-removed-using-quantiles-in-python"
---

Alright, let's dive into outlier removal using quantiles in python, a topic I've spent considerable time grappling with, especially when working on that tricky sensor data project a few years back. I was tasked with cleaning up readings that were often spiked by interference and needed a robust method that wouldn't chop off legitimate, albeit extreme, values. It’s more nuanced than just setting hard limits, and quantiles often proved to be the tool for the job.

The core principle behind using quantiles for outlier detection lies in their ability to divide a dataset into equal parts. In essence, a quantile indicates the point below which a specific proportion of the data falls. For instance, the 0.25 quantile (or the 25th percentile) is the value below which 25% of the data lies. Similarly, the 0.75 quantile (75th percentile) marks the point where 75% of the data sits. This makes them exceptionally useful for identifying data points significantly distant from the 'bulk' of your distribution. Unlike standard deviation, quantiles are less sensitive to extreme values themselves, providing a more robust measure of data dispersion. We're not as affected by those problematic outliers when establishing our baselines, which is a huge advantage when cleaning real-world data.

Now, let's move onto the practical side. Using python, the `numpy` library makes calculating quantiles straightforward. I'll demonstrate three common approaches I've used and seen successfully employed in various contexts.

**Example 1: Simple Upper and Lower Bound Truncation**

The most basic application is to remove outliers using a specified lower and upper quantile. In this scenario, we will filter out any data that falls below the lower quantile or above the upper quantile, essentially creating a "valid" range for our data.

```python
import numpy as np

def remove_outliers_quantile_truncation(data, lower_quantile=0.05, upper_quantile=0.95):
    """Removes outliers by truncating values below the lower quantile and above the upper quantile.

    Args:
        data (array_like): Input data.
        lower_quantile (float): Lower quantile value (0-1). Default is 0.05.
        upper_quantile (float): Upper quantile value (0-1). Default is 0.95.

    Returns:
        numpy.ndarray: Data with outliers removed.
    """
    lower_bound = np.quantile(data, lower_quantile)
    upper_bound = np.quantile(data, upper_quantile)
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

# Example usage
sample_data = np.array([10, 12, 15, 13, 11, 120, 9, 7, 14, -20])
cleaned_data = remove_outliers_quantile_truncation(sample_data)
print(f"Original data: {sample_data}")
print(f"Data after truncation: {cleaned_data}")
```

In this snippet, we compute the specified lower and upper quantiles and then create a new array containing only data points that fall within those bounds. This is the go-to approach when you need a clear and straightforward removal strategy. We're not modifying the data, just selecting only what we deem valid.

**Example 2: Capping Outliers Instead of Removing**

Sometimes completely removing outliers can lead to information loss. A more nuanced tactic is to cap outliers, replacing values that fall outside the quantile range with the corresponding bound. This method retains the number of original data points and mitigates the influence of extreme values without losing potentially valuable data.

```python
import numpy as np

def cap_outliers_quantile(data, lower_quantile=0.05, upper_quantile=0.95):
    """Caps outliers by replacing values below the lower quantile with the lower bound
    and values above the upper quantile with the upper bound.

    Args:
        data (array_like): Input data.
        lower_quantile (float): Lower quantile value (0-1). Default is 0.05.
        upper_quantile (float): Upper quantile value (0-1). Default is 0.95.

    Returns:
        numpy.ndarray: Data with outliers capped.
    """
    lower_bound = np.quantile(data, lower_quantile)
    upper_bound = np.quantile(data, upper_quantile)
    capped_data = np.copy(data)  # Create a copy to avoid modifying the original data
    capped_data[capped_data < lower_bound] = lower_bound
    capped_data[capped_data > upper_bound] = upper_bound
    return capped_data

# Example usage
sample_data = np.array([10, 12, 15, 13, 11, 120, 9, 7, 14, -20])
capped_data = cap_outliers_quantile(sample_data)
print(f"Original data: {sample_data}")
print(f"Data after capping: {capped_data}")
```

Here, we calculate the same lower and upper bounds. Instead of filtering the data, we create a copy and then replace values outside those bounds with the bound values themselves. This approach proves valuable when maintaining the original data's size is critical.

**Example 3: Iterative Quantile Filtering**

For datasets with complex distributions and multiple potential outliers, a single pass might not be enough. I often find that applying quantile filtering iteratively can produce significantly cleaner results. This involves repeatedly calculating quantiles and removing or capping outliers until the distribution stabilizes. While computationally intensive, it can be worth it when dealing with real-world data.

```python
import numpy as np

def iterative_quantile_filter(data, lower_quantile=0.05, upper_quantile=0.95, iterations=3):
    """Applies quantile-based filtering iteratively.

    Args:
        data (array_like): Input data.
        lower_quantile (float): Lower quantile value (0-1). Default is 0.05.
        upper_quantile (float): Upper quantile value (0-1). Default is 0.95.
        iterations (int): The number of filtering iterations. Default is 3.

    Returns:
        numpy.ndarray: Data with outliers removed after iterative filtering.
    """
    filtered_data = np.copy(data)
    for _ in range(iterations):
        lower_bound = np.quantile(filtered_data, lower_quantile)
        upper_bound = np.quantile(filtered_data, upper_quantile)
        filtered_data = filtered_data[(filtered_data >= lower_bound) & (filtered_data <= upper_bound)]
        if filtered_data.size == 0:  # Handle the case where the filter completely removes all data.
            return np.array([])
    return filtered_data

# Example usage
sample_data = np.array([10, 12, 15, 13, 11, 120, 9, 7, 14, -20, 150, -30])
filtered_data = iterative_quantile_filter(sample_data)
print(f"Original data: {sample_data}")
print(f"Data after iterative filtering: {filtered_data}")
```

This example uses the initial truncation method and applies it multiple times. On each iteration, we recalculate the quantiles using the *filtered* data from the previous pass, allowing the bounds to adjust to the more refined data distribution. This approach allows for fine-tuning the removal of outliers by repeatedly applying our filter which can be valuable for complex distributions and data cleaning scenarios. I've found this effective in situations where a single application still leaves undesirable edge cases.

**Considerations and Further Exploration**

While quantiles are a great tool, they're not a one-size-fits-all solution. When deciding on the right quantiles, a good starting point would be 0.05 and 0.95 as seen in the provided snippets, but these will depend on the nature of your data. Lower values will result in a greater proportion of data considered outliers, while higher values will be more conservative.

For a deeper theoretical understanding, I recommend exploring books like "All of Statistics" by Larry Wasserman. It covers the theoretical foundations of statistical methods, which are crucial for understanding *why* these techniques work, not just *how*. Additionally, “Statistical Methods for Handling Outliers” by David M. Hawkins provides more in-depth considerations of the detection and handling of outliers, going beyond just quantiles, and covering more advanced methods. Understanding the nature of the distribution of your data is critical for choosing an appropriate technique and quantile values. These resources helped immensely when I was tackling data cleaning problems, and I believe they could be invaluable for anyone working with similar issues. The best practice, as always, is to apply critical analysis and visualization throughout the entire process to ensure that our manipulations aren't skewing the data in an unintended way. It's a matter of balance—removing noise while retaining essential insights.
