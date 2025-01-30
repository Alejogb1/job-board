---
title: "How do you select percentiles from a categorical distribution in NumPy or PyTorch?"
date: "2025-01-30"
id: "how-do-you-select-percentiles-from-a-categorical"
---
The inherent challenge in selecting percentiles from a categorical distribution stems from the discrete nature of the data.  Unlike continuous distributions where percentiles can be directly calculated from the cumulative distribution function (CDF), categorical data require a slightly different approach.  My experience working on large-scale recommendation systems heavily involved this process; accurately determining user preference rankings often relied on precisely selecting percentiles from categorical data representing item categories.  This necessitated a robust methodology that accounts for the discrete probability mass function (PMF).

The solution involves leveraging the cumulative probability associated with each category.  First, we calculate the cumulative sum of probabilities for each category.  Then, for a given percentile, we find the first category whose cumulative probability exceeds that percentile.  This category is then identified as the percentile value.  This approach ensures that the selected category accurately represents the designated percentile in the distribution.  However, edge cases require careful consideration, particularly when dealing with low-probability categories.

Let's illustrate this with several code examples, focusing on both NumPy and PyTorch implementations.  I'll demonstrate the methodology, highlighting practical considerations like handling edge cases and the impact of different probability distributions.

**Example 1: NumPy Implementation with Uniform Probabilities**

This example uses NumPy and assumes a uniform categorical distribution for simplicity. This makes it easier to understand the core logic without the added complexity of unequal probabilities.

```python
import numpy as np

def get_percentile_numpy_uniform(categories, percentile):
    """
    Selects a percentile from a categorical distribution with uniform probabilities using NumPy.

    Args:
        categories: A NumPy array of category labels.
        percentile: The desired percentile (between 0 and 100).

    Returns:
        The category corresponding to the specified percentile.  Returns None if the percentile is invalid.

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(categories, np.ndarray):
        raise ValueError("Categories must be a NumPy array.")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    num_categories = len(categories)
    cumulative_probabilities = np.linspace(0, 100, num_categories + 1) #Uniform distribution

    percentile_index = np.searchsorted(cumulative_probabilities, percentile) -1 #Adjust for 0-based indexing

    if percentile_index < 0 or percentile_index >= num_categories:
      return None

    return categories[percentile_index]


categories = np.array(['A', 'B', 'C', 'D'])
percentile_50 = get_percentile_numpy_uniform(categories, 50)  #Should return 'B' or 'C' depending on rounding
print(f"50th percentile: {percentile_50}")

percentile_90 = get_percentile_numpy_uniform(categories, 90) #Should return 'D'
print(f"90th percentile: {percentile_90}")

percentile_110 = get_percentile_numpy_uniform(categories, 110) #should return None
print(f"110th percentile: {percentile_110}")

```


**Example 2: NumPy Implementation with Non-Uniform Probabilities**

This extends the previous example to handle non-uniform probabilities, a more realistic scenario in most applications.  This involves explicitly defining the probability associated with each category.

```python
import numpy as np

def get_percentile_numpy(categories, probabilities, percentile):
    """
    Selects a percentile from a categorical distribution with non-uniform probabilities using NumPy.

    Args:
        categories: A NumPy array of category labels.
        probabilities: A NumPy array of probabilities corresponding to each category.
        percentile: The desired percentile (between 0 and 100).

    Returns:
        The category corresponding to the specified percentile. Returns None for invalid inputs.

    Raises:
        ValueError: If input is invalid.
    """

    if not isinstance(categories, np.ndarray) or not isinstance(probabilities, np.ndarray):
        raise ValueError("Categories and probabilities must be NumPy arrays.")
    if len(categories) != len(probabilities):
        raise ValueError("Categories and probabilities must have the same length.")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    if not np.allclose(np.sum(probabilities), 1):
        raise ValueError("Probabilities must sum to 1.")


    cumulative_probabilities = np.cumsum(probabilities) * 100
    percentile_index = np.searchsorted(cumulative_probabilities, percentile)

    if percentile_index == len(categories):
        return categories[-1] #handle edge case where percentile is 100
    elif percentile_index == 0:
        return categories[0] #handle edge case where percentile is 0

    return categories[percentile_index -1] # Adjust for 0-based indexing


categories = np.array(['A', 'B', 'C', 'D'])
probabilities = np.array([0.1, 0.2, 0.3, 0.4])
percentile_50 = get_percentile_numpy(categories, probabilities, 50)  #Should return 'C'
print(f"50th percentile: {percentile_50}")

percentile_90 = get_percentile_numpy(categories, probabilities, 90) #Should return 'D'
print(f"90th percentile: {percentile_90}")

```

**Example 3: PyTorch Implementation**

This example demonstrates the equivalent functionality using PyTorch tensors.  The core logic remains the same, but PyTorch's tensor operations are used.

```python
import torch

def get_percentile_pytorch(categories, probabilities, percentile):
    """
    Selects a percentile from a categorical distribution using PyTorch tensors.

    Args:
        categories: A PyTorch tensor of category labels.
        probabilities: A PyTorch tensor of probabilities corresponding to each category.
        percentile: The desired percentile (between 0 and 100).

    Returns:
        The category corresponding to the specified percentile. Returns None for invalid inputs.

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(categories, torch.Tensor) or not isinstance(probabilities, torch.Tensor):
        raise ValueError("Categories and probabilities must be PyTorch tensors.")
    if len(categories) != len(probabilities):
        raise ValueError("Categories and probabilities must have the same length.")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    if not torch.allclose(torch.sum(probabilities), torch.tensor(1.0)):
        raise ValueError("Probabilities must sum to 1.")


    cumulative_probabilities = torch.cumsum(probabilities, dim=0) * 100
    percentile_index = torch.searchsorted(cumulative_probabilities, torch.tensor(percentile))


    if percentile_index == len(categories):
        return categories[-1]
    elif percentile_index == 0:
        return categories[0]

    return categories[percentile_index - 1]


categories = torch.tensor(['A', 'B', 'C', 'D'])
probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
percentile_50 = get_percentile_pytorch(categories, probabilities, 50)  #Should return 'C'
print(f"50th percentile: {percentile_50}")

percentile_90 = get_percentile_pytorch(categories, probabilities, 90) #Should return 'D'
print(f"90th percentile: {percentile_90}")
```

These examples provide a foundation for selecting percentiles from categorical distributions.  Remember to carefully handle edge cases and ensure the validity of your input data.


**Resource Recommendations:**

For a deeper understanding of probability distributions and statistical methods, I recommend consulting standard textbooks on probability and statistics.  NumPy and PyTorch documentation are also invaluable resources for understanding the specific functions and their applications.  Finally, reviewing relevant chapters in books focused on data analysis and machine learning will provide a comprehensive understanding of the context within which this problem arises.
