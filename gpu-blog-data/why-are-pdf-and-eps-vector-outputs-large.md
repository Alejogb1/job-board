---
title: "Why are PDF and EPS vector outputs large when using mplcairo?"
date: "2025-01-30"
id: "why-are-pdf-and-eps-vector-outputs-large"
---
The observed size increase in PDF and EPS vector outputs generated with mplcairo, compared to other backends like the standard Matplotlib PDF backend, often stems from how mplcairo renders and embeds graphical elements. Specifically, the issue lies within its handling of text and its reliance on Cairo's path-based rendering for nearly all primitives.

The core distinction boils down to the *primitive encoding*. Traditional backends, for example, often encode text as text strings within the PDF or EPS file, leveraging the native font resources of the viewer application. This means the file only contains the text itself, font name, and styling information. Conversely, mplcairo, due to its Cairo integration, typically converts each glyph into a series of vector paths. These paths are effectively the *outlines* of the characters. This approach ensures cross-platform consistency and avoids font-related issues, but it also bloats the file size significantly, particularly for plots with numerous text labels, annotations, or axes tick marks. I encountered this frequently while developing scientific visualization tools where precise font rendering was paramount but resulted in unexpectedly large output files.

My initial experience involved plotting complex multi-panel figures with numerous annotations. Using the standard `matplotlib.backends.backend_pdf` backend, the resulting PDF files were relatively small, typically under a megabyte. Switching to `mplcairo`, aiming for better anti-aliasing and consistency across platforms, I found that the file sizes jumped to several megabytes for the same figure. Upon closer inspection of the PDF content, it became clear that each text character was represented by a complex sequence of paths rather than text strings. This was especially pronounced when using LaTeX rendering which resulted in even more complex glyphs.

Further investigation revealed that this behavior wasn't limited to text. While Matplotlib backends may encode certain graphical elements like rectangles or circles as dedicated primitives within the PDF or EPS file structure, mplcairo often renders these as paths too. Although more precise and faithful to the rendering, this pathway encoding results in increased information that constitutes the file output. Every line, arc, curve and other geometry must be constructed from series of points that must be stored. While this method increases the consistency, and increases the faithfulness to pixel output, it unfortunately also increases file size output considerably. The tradeoff must be considered in every use case of the mplcairo backend.

Here are three code examples that demonstrate the size differences, and offer some solutions to mitigate them.

**Example 1: Simple Text Plot**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import mplcairo

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Example 1a: Using standard Matplotlib PDF backend
with PdfPages('standard_text.pdf') as pdf:
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Simple plot with text (standard backend)')
    pdf.savefig()
    plt.close()

# Example 1b: Using mplcairo backend
with PdfPages('mplcairo_text.pdf') as pdf:
    with plt.rc_context({'backend': 'module://mplcairo.base'}):
        plt.figure()
        plt.plot(x,y)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Simple plot with text (mplcairo backend)')
        pdf.savefig()
    plt.close()
```

This first example demonstrates the core issue of text rendering. While both plots look identical, the `mplcairo_text.pdf` will be significantly larger. The standard PDF backend retains the text as actual text objects within the PDF, while the mplcairo output converts it to path-based glyphs.

**Example 2:  Plot with a large number of markers**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import mplcairo

# Generate some data
x = np.random.rand(1000)
y = np.random.rand(1000)


# Example 2a: Using standard Matplotlib PDF backend
with PdfPages('standard_markers.pdf') as pdf:
    plt.figure()
    plt.scatter(x,y, marker='o', s=10)
    plt.title('Scatter plot with many markers (standard backend)')
    pdf.savefig()
    plt.close()

# Example 2b: Using mplcairo backend
with PdfPages('mplcairo_markers.pdf') as pdf:
    with plt.rc_context({'backend': 'module://mplcairo.base'}):
        plt.figure()
        plt.scatter(x,y, marker='o', s=10)
        plt.title('Scatter plot with many markers (mplcairo backend)')
        pdf.savefig()
    plt.close()
```
This example highlights how path conversion also impacts marker output. When plotted as paths, every marker's outline is converted and stored, contributing to size increases relative to standard approaches.

**Example 3:  Reducing mplcairo Output size through rasterization**
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import mplcairo

# Generate some data
x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x)
# Example 3: Reducing size with rasterization
with PdfPages('mplcairo_raster.pdf') as pdf:
    with plt.rc_context({'backend': 'module://mplcairo.base'}):
        fig, ax = plt.subplots()
        ax.plot(x,y, label = 'sine', rasterized=True)
        ax.plot(x,z, label = 'cosine', rasterized=True)
        ax.set_title('Rasterized Plot')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

```

In this example, I demonstrate a key approach to mitigate the file-size increase issue within mplcairo. Setting `rasterized=True` for relevant plot elements will render those elements as bitmaps and reduce the vector complexity of output PDF. In such rasterization, the data will be saved as a bitmap and not paths, so the output file will become much smaller than the corresponding vector counterparts. In cases like this, a good compromise can often be found where rasterizing complicated plots will significantly reduce the output filesize, while retaining vector paths for axis labels and text.
While rasterization can reduce file sizes, its tradeoff is that zoom resolution of rasterized areas is limited to the DPI of the generated raster image. Care must be taken in each particular use case to select an appropriate resolution.

To effectively manage PDF and EPS output with mplcairo, several approaches can be considered. First, if high-resolution text is not a paramount requirement, the standard backends may be preferred. However, when precise glyph rendering or platform consistency is crucial, the size issue should be mitigated using techniques like selective rasterization. Specifically, applying rasterization to complex plots or scatter plots can often significantly reduce output size, while text can remain as a vectorized component. Additionally, where possible, simplification of plots by reducing the complexity of graphical elements or number of markers can also lead to size reductions. Furthermore, careful thought should be given to which portions of a figure need to be vectorized and which can be rasterized without appreciable loss of quality.

Finally, I recommend a few resources for deepening understanding in this area. Consult the Matplotlib documentation for specific options related to figure output and backend configuration. The Cairo graphics library documentation can provide detailed understanding of the underlying mechanisms for path-based graphics rendering. Furthermore, research into rasterization and image embedding within PDF and EPS formats will help you understand how to trade-off visual fidelity and output size. These resources will provide a comprehensive understanding of the factors at play when dealing with mplcairo and vector graphics output.
