---
title: "How can multiple arrays/images be plotted on a single matplotlib canvas in Python?"
date: "2025-01-30"
id: "how-can-multiple-arraysimages-be-plotted-on-a"
---
The core challenge in plotting multiple arrays or images on a single matplotlib canvas lies in managing subplot arrangements and properly assigning data to these distinct plotting areas. I've encountered this frequently in my work processing multi-spectral satellite imagery, where displaying several bands alongside derived indices is crucial for analysis. Effectively, we need to think of the canvas as a grid, and each array or image becomes a distinct tile within that grid.

To achieve this, we primarily use the `matplotlib.pyplot` module, specifically the `subplots()` and `imshow()` (or `plot()`, `scatter()`, etc., depending on the data type) functions. `subplots()` is our key tool for generating the figure and its axes (the grid tiles), allowing us to specify the number of rows and columns for the layout. Once we have these axes objects, we can selectively target each to display its corresponding data.

Let’s consider a simple scenario: plotting three 1D arrays representing different sensor readings over time. I've often used this approach to visualize time series data from various environmental monitoring stations.

**Code Example 1: Plotting Three 1D Arrays**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
time = np.arange(0, 10, 0.1)
sensor1 = 2 * np.sin(time) + 1
sensor2 = 0.5 * time + np.random.normal(0, 0.5, len(time))
sensor3 = np.cos(time) * np.exp(-time / 5)

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))

# Plot each sensor reading on its designated subplot
axes[0].plot(time, sensor1, label='Sensor 1')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[1].plot(time, sensor2, label='Sensor 2')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[2].plot(time, sensor3, label='Sensor 3')
axes[2].set_ylabel('Amplitude')
axes[2].set_xlabel('Time')
axes[2].legend()

# Adjust layout to prevent overlaps and display plot
plt.tight_layout()
plt.show()
```

In this example, `subplots(nrows=3, ncols=1)` creates a figure with three rows and one column of subplots, effectively creating three separate plots stacked vertically. The function returns two variables: `fig`, the figure object representing the entire canvas, and `axes`, an array of individual axes objects representing each subplot. I then use array indexing (`axes[0]`, `axes[1]`, `axes[2]`) to select the correct axis for each `plot()` call. The `.set_ylabel()` calls add descriptive labels to each plot's Y-axis, while the `.set_xlabel()` in the last subplot label its X-axis, and `plt.tight_layout()` ensures everything fits nicely, avoiding label overlap. This structure is particularly useful when comparing measurements from multiple sources, a common task in my work with ecological data.

Now, shifting our focus to images, a slightly different method is required. We use `imshow()` to display the array as an image, taking special consideration of color maps. I've frequently dealt with grayscale and false-color imaging in remote sensing.

**Code Example 2: Plotting Three 2D Arrays as Images**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample image data
image1 = np.random.rand(50, 50)
image2 = np.random.rand(50, 50) * 2 # scaling to show color variations
image3 = np.random.rand(50, 50) * 3

# Create a figure and set of subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

# Display each image on its designated subplot
axes[0].imshow(image1, cmap='gray')
axes[0].set_title('Image 1')
axes[1].imshow(image2, cmap='viridis')
axes[1].set_title('Image 2')
axes[2].imshow(image3, cmap='magma')
axes[2].set_title('Image 3')

# Remove axis ticks
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout and display
plt.tight_layout()
plt.show()
```

Here, `subplots(nrows=1, ncols=3)` generates three subplots arranged horizontally.  The key difference is the use of `imshow()`, which interprets the 2D arrays as image data, with `cmap` specifying the color mapping to use (in this case, gray scale, viridis and magma). The scaling applied to `image2` and `image3` demonstrates that changes in array values get visualized by the selected color maps. I’ve omitted ticks in this case, as images usually do not require them, adding the titles to identify each image. This structure allows side-by-side comparison of different processing results, which is very relevant in my workflow, especially for comparing pre- and post-processing outputs.

A more complex scenario involves combining both 1D line plots and 2D image plots on the same canvas.  This can be useful for juxtaposing temporal data with spatial patterns, such as showing a single sensor's response while simultaneously displaying its location relative to the environment.

**Code Example 3: Combining 1D and 2D Plots**

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample 1D and 2D data
time = np.arange(0, 10, 0.1)
sensor_data = np.sin(time) + 0.2 * np.random.normal(0, 1, len(time))
spatial_data = np.random.rand(50, 50)


# Create figure and subplots with a flexible arrangement
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))


# Flatten the array of axes for ease of indexing
axes = axes.flatten()

# Plot the sensor data on the first subplot
axes[0].plot(time, sensor_data, label='Sensor Reading')
axes[0].set_ylabel('Amplitude')
axes[0].set_xlabel('Time')
axes[0].legend()


# Display the spatial data as an image on the second subplot
axes[1].imshow(spatial_data, cmap='viridis')
axes[1].set_title('Spatial Data')
axes[1].set_xticks([])
axes[1].set_yticks([])


# Adding a simple text subplot as placeholder in the third subplot
axes[2].text(0.5, 0.5, 'Additional Info Placeholder', ha='center', va='center', fontsize=12)
axes[2].set_xticks([])
axes[2].set_yticks([])

# Hide the unused fourth subplot
fig.delaxes(axes[3])


# Adjust layout and display
plt.tight_layout()
plt.show()
```

In this example, `subplots(nrows=2, ncols=2)` creates a 2x2 grid of subplots.  I use `.flatten()` to convert the 2D axes array to a 1D array for easier access via indexing. I then place the 1D plot in `axes[0]`, the image in `axes[1]`, and a dummy text plot in `axes[2]`, using code similar to the previous examples. Finally, the unused subplot is removed using `fig.delaxes(axes[3])`, and `plt.tight_layout()` and `plt.show()` handle the formatting and display. The flexibility here allows me to juxtapose various data types within one visual context, something I often find useful when building detailed environmental assessments.

In summary, effectively using `subplots()` along with the appropriate plotting or imaging function (like `plot()`, `scatter()`, `imshow()`) is key to plotting multiple arrays or images on a single matplotlib canvas. Managing the axes objects correctly and adjusting layout using `plt.tight_layout()` ensures that information is presented in a clear and concise manner.

For further learning, I recommend exploring the official Matplotlib documentation, particularly focusing on the sections dealing with figures, subplots, and plotting/imaging functions. Books on scientific data visualization using Python are also valuable resources. Lastly, reviewing online examples within the matplotlib community can provide different perspectives and use cases.
