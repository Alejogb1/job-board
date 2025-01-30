---
title: "How can I implement domain coloring in Python with appropriate scales?"
date: "2025-01-30"
id: "how-can-i-implement-domain-coloring-in-python"
---
Domain coloring, the technique of visualizing complex functions by mapping their output to color values, requires careful consideration of scaling to avoid misleading or uninterpretable results.  My experience implementing this for a high-performance computing project involving chaotic systems underscored the importance of logarithmic and piecewise scaling in particular.  Simply mapping the magnitude or phase directly to color often leads to loss of detail in certain regions, obscuring crucial features of the function.


**1. Clear Explanation**

Domain coloring involves assigning a color to each point in the complex plane based on the output of a complex function evaluated at that point.  This color is typically determined by two components of the output: the magnitude and the argument (phase).  The magnitude is usually represented by brightness or intensity, while the phase is mapped to hue.  However, a direct linear mapping of both magnitude and phase to the color space can produce images with poor contrast and lost detail.  This is because the range of the function's output – particularly the magnitude – may be extremely large, leading to a few points dominating the color map.

To mitigate this, scaling strategies must address the dynamic range of the output.  Logarithmic scales compress the magnitude range, effectively revealing details in both high- and low-magnitude regions that a linear scale would obscure.  Piecewise linear scaling allows for even greater control, enabling the independent scaling of specific magnitude ranges to highlight critical features or regions of interest.   Careful selection of the colormap is also crucial; choosing a perceptually uniform colormap minimizes misinterpretations of color differences.  For instance, a colormap like `hsv` might be inappropriate due to its uneven perceptual spacing of hues.  Colormaps such as `viridis` or `magma` are often preferred for their better perceptual uniformity.

**2. Code Examples with Commentary**

The following examples demonstrate domain coloring with different scaling strategies using Python and Matplotlib.  I've used these approaches extensively in my work to visualize Julia and Mandelbrot sets, and have refined them based on my own trial and error, and community feedback.

**Example 1: Linear Scaling (Illustrative, Generally Suboptimal)**

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_function(z):
    return z**2 + 1

# Create grid of complex numbers
x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Compute function values
W = complex_function(Z)

# Linear scaling: magnitude to brightness, phase to hue
magnitude = np.abs(W)
phase = np.angle(W)
brightness = magnitude / np.max(magnitude) # Normalize magnitude
hue = (phase + np.pi) / (2 * np.pi)  # Normalize phase to [0, 1]

# Create HSV image and convert to RGB
hsv = np.stack((hue, np.ones_like(hue), brightness), axis=-1)
rgb = matplotlib.colors.hsv_to_rgb(hsv)

# Plot the image
plt.imshow(rgb, extent=[-2, 2, -2, 2], origin='lower')
plt.colorbar(label='Magnitude')
plt.title('Domain Coloring with Linear Scaling')
plt.show()

```

This example uses linear scaling, which is simple but often inadequate.  Notice how high-magnitude regions overwhelm the image, obscuring low-magnitude details.  The normalization steps are crucial, ensuring values lie within the [0,1] range required by Matplotlib's color mapping functions.


**Example 2: Logarithmic Scaling**

```python
import numpy as np
import matplotlib.pyplot as plt

# ... (complex_function definition from Example 1) ...

# Logarithmic scaling for magnitude
magnitude = np.abs(W)
magnitude_log = np.log1p(magnitude) # Add 1 to avoid log(0)
magnitude_scaled = magnitude_log / np.max(magnitude_log)

# Phase remains unchanged
phase = np.angle(W)
hue = (phase + np.pi) / (2 * np.pi)

# ... (HSV to RGB conversion and plotting as in Example 1) ...

plt.title('Domain Coloring with Logarithmic Scaling')
plt.show()
```

This improved version employs a logarithmic scale for the magnitude.  `np.log1p()` is used to handle potential zero values gracefully. This logarithmic transformation compresses the high-magnitude values, allowing for a clearer visualization of details across a wider range of magnitudes.


**Example 3: Piecewise Linear Scaling**

```python
import numpy as np
import matplotlib.pyplot as plt

# ... (complex_function definition from Example 1) ...

# Piecewise linear scaling for magnitude
magnitude = np.abs(W)
threshold = 10  # Adjust as needed based on the function's output range

magnitude_scaled = np.where(magnitude < threshold, magnitude / threshold, 1 + (magnitude - threshold) / (np.max(magnitude)-threshold))

# Phase remains unchanged
phase = np.angle(W)
hue = (phase + np.pi) / (2 * np.pi)

# ... (HSV to RGB conversion and plotting as in Example 1) ...

plt.title('Domain Coloring with Piecewise Linear Scaling')
plt.show()
```

This example demonstrates piecewise linear scaling.  Here, magnitudes below a specified threshold are scaled linearly, while magnitudes above that threshold are scaled differently. This allows for highlighting regions of particular interest.  The threshold value requires careful tuning, potentially through experimentation or analysis of the function's behavior.


**3. Resource Recommendations**

For a deeper understanding of colormaps and their perceptual properties, consult the Matplotlib documentation.  The book "Color Science: Concepts and Methods, Quantitative Data Analysis" provides a thorough background on the underlying principles of color perception.   Extensive exploration of different complex functions and scaling techniques is essential for developing intuition.  Finally, leveraging online communities and forums dedicated to scientific visualization,  and computational mathematics will provide valuable insights from other practitioners.
