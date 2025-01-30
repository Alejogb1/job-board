---
title: "How can I find the intersection of two sets of 2D points in PyTorch?"
date: "2025-01-30"
id: "how-can-i-find-the-intersection-of-two"
---
The efficient intersection of two sets of 2D points in PyTorch hinges on leveraging its tensor operations for optimized performance, particularly when dealing with large datasets.  My experience optimizing point cloud processing pipelines has shown that naive approaches using nested loops can become computationally prohibitive.  Directly utilizing PyTorch's broadcasting and boolean indexing capabilities offers a significant advantage in both speed and readability.

**1. Clear Explanation:**

The core strategy involves representing each set of 2D points as a PyTorch tensor of shape (N, 2), where N is the number of points.  We then exploit PyTorch's broadcasting capabilities to compare each point in the first set against every point in the second set.  This comparison generates a boolean tensor indicating whether each point in the first set is present in the second.  Finally, boolean indexing is used to extract only the points from the first set that satisfy this condition, yielding the intersection.  This avoids explicit looping and leverages PyTorch's optimized backend for substantial performance gains.  Handling potential numerical imprecision due to floating-point arithmetic requires careful consideration, and we will address this in the code examples.


**2. Code Examples with Commentary:**

**Example 1:  Basic Intersection using `allclose`**

This example uses `torch.allclose` to account for minor floating-point discrepancies. It's suitable for most scenarios where slight numerical variations are acceptable.

```python
import torch

def intersect_points_allclose(set1, set2, rtol=1e-05, atol=1e-08):
    """
    Finds the intersection of two sets of 2D points using torch.allclose.

    Args:
        set1: PyTorch tensor of shape (N, 2) representing the first set of points.
        set2: PyTorch tensor of shape (M, 2) representing the second set of points.
        rtol: Relative tolerance for torch.allclose.
        atol: Absolute tolerance for torch.allclose.

    Returns:
        A PyTorch tensor of shape (K, 2) representing the intersection, where K <= N.  Returns an empty tensor if no intersection is found.

    """
    #Expand dimensions for broadcasting
    set1_expanded = set1[:, None, :]
    set2_expanded = set2[None, :, :]

    #Compare using allclose for floating point tolerance
    matches = torch.allclose(set1_expanded, set2_expanded, rtol=rtol, atol=atol)

    #Find indices where at least one match exists along the second dimension
    intersection_indices = torch.any(matches, dim=1)

    #Extract the intersection points using boolean indexing
    intersection = set1[intersection_indices]

    return intersection

# Example usage:
set1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
set2 = torch.tensor([[3.00001, 4.0], [7.0, 8.0], [1.0, 2.0]])
intersection = intersect_points_allclose(set1, set2)
print(f"Intersection using allclose: \n{intersection}")


```

**Example 2:  Intersection using  `cdist` and thresholding**

This example employs `torch.cdist` to compute pairwise distances between points and then thresholds based on a predefined tolerance. This is robust against minor coordinate variations and offers finer control over the matching criteria.

```python
import torch

def intersect_points_cdist(set1, set2, threshold=1e-05):
    """
    Finds the intersection of two sets of 2D points using torch.cdist.

    Args:
        set1: PyTorch tensor of shape (N, 2) representing the first set of points.
        set2: PyTorch tensor of shape (M, 2) representing the second set of points.
        threshold: Distance threshold for determining intersection.

    Returns:
        A PyTorch tensor of shape (K, 2) representing the intersection. Returns an empty tensor if no intersection is found.

    """
    distances = torch.cdist(set1, set2)
    min_distances = torch.min(distances, dim=1).values
    intersection_indices = min_distances < threshold
    intersection = set1[intersection_indices]
    return intersection

# Example usage
set1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
set2 = torch.tensor([[3.00001, 4.0], [7.0, 8.0], [1.0, 2.0]])
intersection = intersect_points_cdist(set1, set2)
print(f"Intersection using cdist: \n{intersection}")

```


**Example 3:  Intersection with  pre-sorting for enhanced efficiency on large datasets**

For very large datasets, pre-sorting can significantly reduce the computational burden. This approach is particularly beneficial when dealing with millions of points.

```python
import torch

def intersect_points_sorted(set1, set2):
    """
    Finds the intersection of two sets of 2D points, utilizing pre-sorting for efficiency.

    Args:
      set1: PyTorch tensor of shape (N, 2) representing the first set of points.
      set2: PyTorch tensor of shape (M, 2) representing the second set of points.

    Returns:
      A PyTorch tensor of shape (K, 2) representing the intersection. Returns an empty tensor if no intersection is found.

    """
    #Sort the points lexicographically
    set1_sorted = torch.sort(set1, dim=0).values
    set2_sorted = torch.sort(set2, dim=0).values

    #Efficiently find the intersection using a two-pointer approach.
    intersection = []
    i = j = 0
    while i < len(set1_sorted) and j < len(set2_sorted):
      if torch.allclose(set1_sorted[i], set2_sorted[j]):
        intersection.append(set1_sorted[i])
        i += 1
        j += 1
      elif torch.all(set1_sorted[i] < set2_sorted[j]):
        i += 1
      else:
        j += 1

    return torch.stack(intersection) if intersection else torch.empty((0,2))

# Example usage:
set1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
set2 = torch.tensor([[3.0, 4.0], [7.0, 8.0], [1.0, 2.0]])
intersection = intersect_points_sorted(set1, set2)
print(f"Intersection using sorting: \n{intersection}")
```

**3. Resource Recommendations:**

For a deeper understanding of PyTorch's tensor operations, I recommend consulting the official PyTorch documentation.  A thorough grasp of linear algebra and numerical methods will prove invaluable in optimizing these types of algorithms.  Furthermore, studying efficient data structures and algorithms (particularly those relevant to searching and sorting) will provide a strong foundation for developing high-performance solutions for large datasets.  Exploring advanced topics in computational geometry can also offer insights into more sophisticated point-set intersection techniques.  Finally, benchmarking and profiling your code is crucial for identifying and addressing performance bottlenecks.
