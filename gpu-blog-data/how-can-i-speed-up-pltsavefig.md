---
title: "How can I speed up `plt.savefig`?"
date: "2025-01-30"
id: "how-can-i-speed-up-pltsavefig"
---
The bottleneck in `matplotlib.pyplot.savefig` often arises from the inherent complexity of rendering vector graphics, particularly when dealing with high-resolution images or intricate plot elements. I've observed this firsthand during intensive data visualization projects, where generating hundreds of figures per script was commonplace. A primary contributing factor is the default reliance on rasterization, even when saving to vector formats like PDF or SVG; this occurs because Matplotlib rasterizes elements (e.g., scatter plots, complex filled areas) by default for performance when displaying the plot on the screen, and then reuses that rasterization for save operations. This is particularly problematic with large datasets, as pixel data increases significantly. To circumvent this, one must actively manage the drawing process and leverage the capabilities of the backend being used.

The first and often most effective strategy is to ensure that the target format is fully exploiting vector graphics rather than rasterizing intermediate steps. This is controlled, to a significant extent, by Matplotlib's rendering process for the backend. For example, if saving a PDF, we need to ensure that text, lines, and basic shapes are drawn as vector objects rather than rasterized images embedded within the PDF. This is often achieved by explicitly setting the `rasterized=False` keyword argument for relevant plot elements. However, blindly applying this parameter can lead to performance degradation in some scenarios if a plot has many overlapping elements, because then the vector graphics become very complex. There must be a balance and one must test what performs best in specific cases.

Another, often overlooked, area for performance optimization is how the image is saved within the plot's overall structure. Matplotlib, by default, will render the plot into an in-memory buffer (primarily if using raster output formats like PNG or JPG), which is then written to disk. This operation incurs overhead, so it's advantageous to control this process as efficiently as possible. Specifically, using a `bbox_inches` parameter to tighten the plot's boundaries within the file has been shown to reduce the amount of data that needs to be written to disk. If the bounding box encompasses extraneous white space, these are often represented by actual data points in raster formats and this results in unnecessary overhead.

Finally, employing a backend that's optimized for speed or memory usage, or both, can dramatically improve saving time. The default backend is often adequate for simple plots, but when dealing with complex visualizations, switching to a more performant backend can result in a notable difference. For instance, the Agg backend, which is used for raster output formats (like PNG and JPG) is a good starting point for speed and memory efficiency. Also, for vector output, such as SVG, you should be using the 'SVG' backend. Other alternatives exist, for instance, using Cairo to render and save can be beneficial on Linux-based systems or if one has it installed on Windows or macOS, especially when dealing with anti-aliased lines and complex text rendering. The selection of the backend will often depend on the specifics of the plot and the user's individual platform configuration.

Here are three code examples to illustrate these concepts:

**Example 1: Avoiding Rasterization of Lines and Text**

This example showcases the impact of controlling the rasterization of individual plot elements. The main issue comes into play with complex plots containing many lines. When `rasterized=True` for a line collection or when saving to raster formats, it might result in rasterized lines when using vector output such as PDF or SVG. This operation can be relatively costly, particularly with large amounts of data.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# Create the plot
fig, ax = plt.subplots()
lines = ax.plot(x, y, linewidth=0.5) # Using ax.plot which returns a list of Line2D objects.

#Save with default rasterization
fig.savefig('rasterized_plot.pdf', bbox_inches='tight')

# Set lines not to rasterize.
for line in lines:
  line.set_rasterized(False)

# Save the plot to PDF without rasterization of the lines
fig.savefig('vectorized_plot.pdf', bbox_inches='tight')
plt.close(fig)

```

In this instance, a sinusoidal curve is generated. The key here is setting `rasterized=False` on a line element to instruct the backend to render the line as a vector object. The 'rasterized\_plot.pdf' file will be larger, particularly when dealing with many more data points, and might not perform as well when opened in an editor, because parts of it are rasterized. The 'vectorized\_plot.pdf' will have all the lines rendered as vector objects, and will perform better, but there are no real benefits if one has a simpler plot. It is important to note that setting `rasterized=False` for large plots may actually cause the save operation to take longer, or for the final file to become very large, as the file might contain a large amount of vector data, depending on what we are plotting. Thus one must be aware of the performance and file size trade-offs.

**Example 2: Efficient Bounding Box Usage**

This example demonstrates the importance of specifying a tight bounding box to reduce the amount of data saved in the final file. This is often particularly noticeable with raster outputs. When one has a plot with large areas of white space around it, these white areas are represented as pixels in raster outputs (JPG, PNG etc.) and can result in larger files and longer saving times. The `bbox_inches='tight'` argument in the `savefig` function allows for this behavior.

```python
import matplotlib.pyplot as plt

# Create a simple scatter plot
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [4, 5, 6])
ax.set_xlim(0, 4)
ax.set_ylim(3, 7)

# Save with a regular bounding box
fig.savefig('loose_bbox.png', dpi=300)

# Save with tight bounding box
fig.savefig('tight_bbox.png', bbox_inches='tight', dpi=300)
plt.close(fig)
```

In the above snippet, a scatter plot is generated with some padding. If you compare the output images 'loose\_bbox.png' and 'tight\_bbox.png', you'll notice the 'tight\_bbox.png' is both smaller in file size and less computationally intensive. This is due to reduced white space in the file and will reduce saving time, especially with high resolution outputs.

**Example 3: Backend Selection**

This example highlights the effect of changing the Matplotlib backend. The 'Agg' backend is used here, primarily due to its speed and memory efficiency and it is usually the one used by default for raster outputs, but it can be useful to set it explicitly as it can be changed through the `matplotlibrc` file. While in some cases using other backends might be more performant on your system or for a particular type of output (e.g. Cairo for specific needs in Linux systems, or 'SVG' for vector output), the 'Agg' backend is generally very efficient. In this case, we will generate 500 scatter plots.

```python
import matplotlib
matplotlib.use('Agg') # Set Agg as backend.

import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
for i in range(500):
  fig, ax = plt.subplots()
  ax.scatter(np.random.rand(100), np.random.rand(100))
  fig.savefig(f'plot_{i}.png', dpi=150)
  plt.close(fig)
end = time.time()
print(f"Total time taken with Agg: {end - start} seconds")
```

Executing this with the Agg backend, in my case, results in notably faster execution compared to another backend that's primarily used for interactive plotting or that requires more overhead when saving figures, such as the 'Qt5Agg' or 'TkAgg' backends when using `matplotlib` with interactive GUIs. Using 'Agg' when one is only generating plots and not displaying them on screen leads to improved performance.

For further information on optimizing `plt.savefig`, I'd recommend reviewing the documentation within the `matplotlib` project, in particular the documentation for the `savefig` function and the backends documentation. Experimentation with different backends and parameter combinations are also crucial to understand their impact on your specific plotting use cases. There are numerous online forums and communities that delve deeper into specific performance issues, and reading through those is also recommended. Also, be aware that the performance of matplotlib might vary across systems, so test the suggested changes on your local setup, for your particular use case.
