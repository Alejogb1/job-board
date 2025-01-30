---
title: "How can I convert SVG to PNG in Python, including custom fonts?"
date: "2025-01-30"
id: "how-can-i-convert-svg-to-png-in"
---
The reliable conversion of SVG to PNG in Python, particularly when preserving custom fonts, hinges on the capability of the chosen library to accurately interpret and render the SVG's font definitions.  Naive approaches often fail due to discrepancies in font rendering engines between the SVG generation environment and the Python environment used for conversion.  My experience working on a large-scale data visualization project highlighted this limitation; initially using a simplistic method resulted in inconsistent and often incorrect font rendering in the resulting PNGs.

The solution requires a more sophisticated approach, leveraging a library that can handle SVG parsing and rendering within a controlled environment.  I found that `CairoSVG` offers the best balance of capability, reliability, and ease of integration for this task.  While other libraries like `svglib` exist, my experience indicates they are less robust in handling complex SVGs, particularly those containing custom fonts embedded via `<font>` elements or referenced externally.

**1. Clear Explanation**

The core process involves three stages:

* **SVG Parsing:** The SVG file is parsed, extracting all vector data, including path definitions and font information.
* **Font Management:** The custom fonts must be made accessible to the rendering engine. This typically involves specifying the font file paths or embedding the font data within the rendering process.
* **Rasterization:** The parsed vector data is rasterized – converted into a pixel-based image – using a suitable rendering engine. This stage is crucial for accurate font rendering.  The resolution of the resulting PNG image needs to be specified to control image quality.


**2. Code Examples with Commentary**

**Example 1: Basic Conversion with System Fonts**

This example demonstrates a basic conversion assuming the SVG utilizes system-installed fonts.  If your custom font is already installed on the system, this approach might suffice.

```python
import cairocffi as cairo
from cairosvg import svg2png

svg_file = "input.svg"
png_file = "output.png"

svg2png(url=svg_file, write_to=png_file, scale=2) # scale doubles the resolution

```

This code utilizes `cairosvg`.  The `scale` parameter is essential; it controls the resolution of the output PNG.  Increasing the scale value improves image quality but increases file size.  Note that this example does not explicitly handle custom fonts; it relies on the system's font rendering capabilities.


**Example 2: Conversion with Embedded Fonts**

This example handles custom fonts embedded within the SVG using a data URL. This method is less preferred due to potential for increased file sizes and compatibility issues.

```python
import cairocffi as cairo
from cairosvg import svg2png
import base64

# ... (Assume 'custom_font.ttf' is your font file) ...

with open("custom_font.ttf", "rb") as f:
    font_data = f.read()
    encoded_font = base64.b64encode(font_data).decode('utf-8')

svg_with_embedded_font = f"""
<svg>
  <defs>
    <font id="customFont" horiz-adv-x="1000">
      <font-face font-family="CustomFont" units-per-em="1000" panose-1="2 0 0 0 0 0 0 0 0 0">
        <font-face-src>
          <font-face-name name="CustomFont"/>
          <font-face-uri>{f"data:application/x-font-ttf;base64,{encoded_font}"}</font-face-uri>
        </font-face-src>
      </font-face>
    </font>
  </defs>
  <text x="100" y="100" font-family="CustomFont" font-size="30">This uses my custom font</text>
</svg>
"""

svg2png(bytestring=svg_with_embedded_font.encode('utf-8'), write_to=png_file, scale=2)

```

This code embeds the custom font data directly into the SVG string before rendering.  The crucial step is encoding the font file as a base64 string and integrating it within the SVG's `<font>` element.  The use of `bytestring` avoids file I/O for this in-memory SVG.  However, this approach can be complex and increases the SVG size significantly.

**Example 3: Conversion with Externally Referenced Fonts**

This example, demonstrating the most robust approach, leverages `CairoSVG`'s capability to handle external font files specified via font paths. This requires ensuring the font file is accessible to the Python environment.

```python
import cairocffi as cairo
from cairosvg import svg2png
import os

svg_file = "input.svg" # This SVG should reference the font using a font-family name
png_file = "output.png"
font_path = "/path/to/custom_font.ttf" # Replace with actual path


os.environ['FONTCONFIG_FILE'] = font_path # Set environment for fontconfig

svg2png(url=svg_file, write_to=png_file, scale=2)

```

This approach uses environment variables to direct the rendering engine to the correct location of the custom font.  The `input.svg` file needs to correctly reference the font family name (e.g., "CustomFont").  Ensure the font path is accurate and the file exists; otherwise, the conversion will fail.  Using environment variables is generally a cleaner and more scalable method than hardcoding paths within the code itself, particularly in larger applications.


**3. Resource Recommendations**

The `CairoSVG` documentation provides comprehensive details on its features and usage.  Familiarizing yourself with the `cairo` library's capabilities will further enhance your understanding of the underlying rendering process.  A good understanding of SVG structure and font embedding mechanisms is vital for successful implementation.  Finally, refer to the documentation of your specific font format (e.g., TrueType, OpenType) for any format-specific considerations.  These resources will significantly aid in troubleshooting and achieving reliable results.
