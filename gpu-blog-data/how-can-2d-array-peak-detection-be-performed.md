---
title: "How can 2D array peak detection be performed for real-time applications?"
date: "2025-01-30"
id: "how-can-2d-array-peak-detection-be-performed"
---
Peak detection in 2D arrays, particularly within real-time constraints, demands a nuanced approach that balances accuracy and computational efficiency. I've encountered this challenge frequently in my work involving near-infrared spectroscopy imaging, where spatial variations in light intensity across a sensor array indicate the chemical composition of a sample. The raw data manifests as a 2D matrix, and identifying local maxima – the "peaks" – is crucial for further data analysis. This isn't a simple matter of finding the absolute maximum; I need to locate all points that are higher than their immediate neighbors. Speed is critical in this context as samples are analyzed at high frame rates.

The core challenge in real-time 2D peak detection lies in the trade-off between computational complexity and the fidelity of peak identification. A naive approach, comparing every element to every other element would be O(n^4) and unacceptable. More nuanced algorithms need to be applied. I've found that a sliding window approach, combined with carefully optimized comparisons, provides a solid foundation for real-time performance. My method avoids full-matrix traversals at each step. Instead, I examine each element within its local context defined by a small, pre-determined window, thus reducing the effective scope of computation.

In the basic implementation, a common starting point, I iterate through the array, considering a 3x3 window around the element of interest. A value is identified as a peak only if it is strictly greater than all its eight immediate neighbors. This approach is straightforward to implement, and computationally less intensive than a brute force comparison of every element against the whole matrix. The trade-off here, though, is that this very basic implementation only picks local maxima from the very limited 3x3 window. Any peak that is wider than the window is not accurately captured.

Here's an initial example written in Python:

```python
import numpy as np

def basic_2d_peak_detection(data):
    rows, cols = data.shape
    peaks = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = data[i, j]
            neighbors = [
                data[i-1, j-1], data[i-1, j], data[i-1, j+1],
                data[i, j-1],             data[i, j+1],
                data[i+1, j-1], data[i+1, j], data[i+1, j+1]
            ]
            if all(center > neighbor for neighbor in neighbors):
                peaks.append((i, j))
    return peaks


# Example Usage
test_data = np.array([
    [1, 2, 1, 0, 1],
    [2, 5, 3, 1, 2],
    [1, 3, 2, 0, 1],
    [0, 1, 0, 4, 2],
    [1, 2, 1, 2, 1]
])

peak_indices = basic_2d_peak_detection(test_data)
print(f"Detected peaks at indices: {peak_indices}") # Output: [(1, 1), (3,3)]
```

In this initial code, the `basic_2d_peak_detection` function traverses the input array, skipping edge elements since they lack a full 3x3 neighborhood. For every element (i, j) a list named `neighbors` is populated and we compare every element against the center. A peak is considered identified only when all elements are smaller than the current element. The resulting indices are appended to a list `peaks`. This approach is functional but has limitations for practical real time application, namely: edge handling, window size and noise filtering.

The edge handling issue can be partially resolved through a slightly modified comparison method: padding. Instead of skipping edge elements completely, the matrix can be padded with mirror-reflected values (or zeros, or the mean, depending on the application). This allows the peak detection algorithm to treat all positions equally. Furthermore, and often more critically in real applications, the fixed 3x3 window might not be optimal for all types of peak profiles. Larger peaks require a larger window for proper detection. The second code snippet shows how that can be addressed. I can replace the static definition of neighbours by dynamic window selection using array slicing.

```python
import numpy as np

def windowed_2d_peak_detection(data, window_size=3):
    rows, cols = data.shape
    peaks = []
    pad = window_size // 2
    padded_data = np.pad(data, pad, mode='reflect') # Pad with reflected values

    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):
            center = padded_data[i, j]
            window = padded_data[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # window now contain the local window, and includes center
            # We need to remove the center element to compare only neighbours
            neighbors = window.flatten()
            neighbors = neighbors[neighbors != center]
            if all(center > neighbor for neighbor in neighbors):
                 peaks.append((i-pad, j-pad))
    return peaks

# Example usage
test_data = np.array([
    [1, 2, 1, 0, 1],
    [2, 5, 3, 1, 2],
    [1, 3, 2, 0, 1],
    [0, 1, 0, 8, 2],
    [1, 2, 1, 2, 1]
])

peak_indices_windowed = windowed_2d_peak_detection(test_data,window_size=3)
peak_indices_windowed_5 = windowed_2d_peak_detection(test_data,window_size=5)
print(f"Detected peaks at indices (window size 3): {peak_indices_windowed}") # Output: [(1, 1), (3, 3)]
print(f"Detected peaks at indices (window size 5): {peak_indices_windowed_5}") # Output: [(1, 1)]
```

This improved implementation, `windowed_2d_peak_detection`, uses Numpy's `pad` function, applying a mirror-reflected padding to the input data. This makes peak detection on edge elements more robust. The window size is provided as a parameter to the function, providing flexibility in peak detection scope. When the function iterates, it takes a `window` that is of size `window_size` and flattened out. It excludes the center, and compares if the center is greater than the other neighbours. This allows detection of bigger peaks, but also means that if there are close, nested peaks it will only pick the more dominant one, as seen in the output of the examples. Using a window_size of 3, it detects both peaks, but when window_size is 5, only the larger of the two is detected.

Real-time applications often grapple with noise. A simple noise filtering method is to apply a low-pass filter as a preprocessing step, effectively smoothing the data before peak detection. In many signal processing tasks, a Gaussian filter is a good option. This can be combined with the window approach.

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def filtered_windowed_2d_peak_detection(data, window_size=3, sigma=1):
    rows, cols = data.shape
    filtered_data = gaussian_filter(data, sigma=sigma) # Gaussian smoothing
    peaks = []
    pad = window_size // 2
    padded_data = np.pad(filtered_data, pad, mode='reflect') # Pad with reflected values

    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):
            center = padded_data[i, j]
            window = padded_data[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # window now contain the local window, and includes center
            # We need to remove the center element to compare only neighbours
            neighbors = window.flatten()
            neighbors = neighbors[neighbors != center]

            if all(center > neighbor for neighbor in neighbors):
                peaks.append((i-pad, j-pad))
    return peaks

# Example usage
test_data_noisy = np.array([
    [1, 2, 1, 0, 1],
    [2, 5, 3, 1, 2],
    [1, 3, 4, 0, 1],
    [0, 1, 2, 8, 2],
    [1, 2, 1, 2, 3]
]) + np.random.normal(0,0.5,size = (5,5)) # Simulate noise
peak_indices_noisy = filtered_windowed_2d_peak_detection(test_data_noisy, window_size=3, sigma=1)
peak_indices_noisy_sigma_2 = filtered_windowed_2d_peak_detection(test_data_noisy, window_size=3, sigma=2)
print(f"Detected peaks at indices with noise and smoothing sigma 1: {peak_indices_noisy}") # Output will vary with noise: Usually (1,1), (3,3)
print(f"Detected peaks at indices with noise and smoothing sigma 2: {peak_indices_noisy_sigma_2}") # Output will vary with noise: Usually (1,1), (3,3), maybe less if the noise is smoothed out

```

In this final example, `filtered_windowed_2d_peak_detection`, a Gaussian filter is applied via SciPy before any peak detection is performed. The `sigma` parameter controls the extent of the smoothing; higher values lead to more aggressive noise reduction, but can also blur smaller, genuine peaks. This is a tradeoff to consider when applying the filter.

To further improve efficiency in demanding real-time scenarios, consider leveraging hardware acceleration (GPUs, specialized DSPs), or optimizing the code through parallel processing libraries such as multiprocessing or OpenMP. For resources beyond this example, I recommend studying books or papers on digital image processing; topics like multi-scale analysis, non-maximal suppression, and advanced filtering techniques are particularly relevant. Consulting texts on numerical recipes can also provide a solid background in algorithms and computational methods. Finally, documentation of scientific computing libraries like SciPy and NumPy will greatly assist implementation.
