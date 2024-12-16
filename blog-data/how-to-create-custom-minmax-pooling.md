---
title: "How to create custom minmax pooling?"
date: "2024-12-16"
id: "how-to-create-custom-minmax-pooling"
---

Okay, let’s unpack custom min-max pooling, something I've found myself needing more than a few times in the trenches. It's not always enough to rely on the standard implementations; sometimes, the specific contours of your problem demand a tailored approach. Instead of simply grabbing the highest or lowest values within a region, you might want to apply some more complex aggregation logic. Let me illustrate.

I remember back at 'InnovateDynamics,' we were processing seismic data, and the signal-to-noise ratio was atrocious. Standard max pooling amplified noise along with signal peaks, while standard min pooling did the opposite. We needed a dynamic pool, one that could, based on surrounding context, effectively reduce both, while still preserving the crucial signal features. Thus began my deep dive into creating custom min-max pooling operations.

The core idea is, frankly, pretty straightforward. You define a window or kernel, and you slide that window across your input data. The magic lies in what happens *inside* that window. Instead of simply taking the max or min, you compute an arbitrary function using all of the values within that window. This flexibility is where the real power lies.

Let’s break down three examples using python and numpy, illustrating different use cases and techniques. In each, we’ll define a function that takes the window’s data and returns a single aggregated value.

**Example 1: Weighted Average Based Pooling**

Suppose we want something akin to a weighted average within each pooling window but are not using standard linear weights. Instead of just a mean or median, let’s say we want to emphasize the center of the window more than the edges, something that would be appropriate in signal processing or image analysis, where the center pixel is often more informative.

```python
import numpy as np

def weighted_avg_pool(window_data):
    """
    Pools the data by taking a weighted average,
    emphasizing the center.
    """
    size = window_data.size
    center = size // 2
    weights = np.exp(-0.5 * ((np.arange(size) - center)**2) / (size / 4)**2 )
    weighted_sum = np.sum(window_data * weights)
    return weighted_sum / np.sum(weights)


def custom_pooling(data, kernel_size, stride, pooling_function):
    """
    Applies a custom pooling function to the data.
    """
    height, width = data.shape
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    output = np.zeros((output_height, output_width))

    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            end_row = start_row + kernel_size
            start_col = col * stride
            end_col = start_col + kernel_size

            window = data[start_row:end_row, start_col:end_col]
            output[row, col] = pooling_function(window)
    return output

# Example Usage:
data = np.random.rand(10, 10)
pooled_data = custom_pooling(data, kernel_size=3, stride=2, pooling_function=weighted_avg_pool)
print("Original data shape:", data.shape)
print("Pooled data shape:", pooled_data.shape)
```

Here, `weighted_avg_pool` applies a gaussian-like weight distribution across the window before averaging. Notice the function, `custom_pooling`, is generalized, accepting a custom function, `pooling_function`, which means any pooling algorithm can be plugged in here.

**Example 2: A "K-th" Smallest Element Pooling**

What if instead of the absolute smallest or largest, we want to pool by choosing the k-th smallest element in the window? This could be useful in edge detection, or to find consistently low-valued regions in, say, atmospheric pressure maps.

```python
import numpy as np

def kth_smallest_pool(window_data, k=2):
    """
    Pools the data by selecting the k-th smallest
    element in the window
    """
    flattened = np.sort(window_data.flatten())
    if k > len(flattened):
        raise ValueError(f"K must be less than or equal to the window size, got k = {k}")
    return flattened[k-1]


def custom_pooling(data, kernel_size, stride, pooling_function, k=2):
    """
    Applies a custom pooling function to the data, with an additional parameter 'k'
    """
    height, width = data.shape
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    output = np.zeros((output_height, output_width))

    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            end_row = start_row + kernel_size
            start_col = col * stride
            end_col = start_col + kernel_size

            window = data[start_row:end_row, start_col:end_col]
            output[row, col] = pooling_function(window,k)
    return output

# Example Usage:
data = np.random.rand(10, 10)
pooled_data = custom_pooling(data, kernel_size=3, stride=2, pooling_function=kth_smallest_pool, k=2)
print("Original data shape:", data.shape)
print("Pooled data shape:", pooled_data.shape)

```

Here, `kth_smallest_pool` takes an extra argument, `k`. We modify `custom_pooling` to pass this argument down. Now we are selecting a very specific data point for pooling. We also introduce a simple error check to ensure that `k` doesn't exceed window size.

**Example 3: Range-Based Pooling**

Another useful approach is to pool based on the *range* (max – min) within the kernel. This is useful when identifying areas with high signal variability, something we used quite a lot when I was dealing with financial data where we wanted to pinpoint volatile regions.

```python
import numpy as np

def range_pool(window_data):
    """
    Pools the data by taking the difference
    between the maximum and the minimum element within the window.
    """
    return np.max(window_data) - np.min(window_data)


def custom_pooling(data, kernel_size, stride, pooling_function):
    """
    Applies a custom pooling function to the data.
    """
    height, width = data.shape
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    output = np.zeros((output_height, output_width))

    for row in range(output_height):
        for col in range(output_width):
            start_row = row * stride
            end_row = start_row + kernel_size
            start_col = col * stride
            end_col = start_col + kernel_size

            window = data[start_row:end_row, start_col:end_col]
            output[row, col] = pooling_function(window)
    return output

# Example Usage:
data = np.random.rand(10, 10)
pooled_data = custom_pooling(data, kernel_size=3, stride=2, pooling_function=range_pool)
print("Original data shape:", data.shape)
print("Pooled data shape:", pooled_data.shape)
```

Here, `range_pool` calculates the range within the window, allowing you to identify areas with high variation.

These examples illustrate the flexibility of custom pooling. You can introduce any logic you require inside the `pooling_function`. This could be any statistical operation, neural network computation, or even a combination of multiple computations.

For delving deeper into this, I would recommend looking into the following:

1.  **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** This book is essential for a comprehensive understanding of the underlying principles of convolutional neural networks (CNNs) and pooling layers. It doesn’t cover custom pooling specifically but provides the foundational knowledge required to implement them effectively.

2.  **Research papers on "Custom CNN Architectures"**: Specific papers relating to architecture search and modified convolution-pooling configurations are very useful. A good place to start your search is google scholar. Keyword searches such as "Adaptive pooling techniques" or "Custom convolution kernels" can quickly point to cutting edge research.

3.  **The Numpy documentation:** I'd suggest spending a significant amount of time in the official numpy docs. It covers everything you need to know to efficiently process matrix-based data in Python, which is crucial when implementing your own pooling operations.

Remember, custom pooling isn't always necessary, but when the standard operations fall short, it can provide the means to precisely extract features your application requires. The examples above, as straightforward as they are, form the basis for far more sophisticated implementations. The key is to always think of your desired output and then craft a function that processes the data accordingly, within a sliding window.
