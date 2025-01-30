---
title: "How can PDF file size be optimized?"
date: "2025-01-30"
id: "how-can-pdf-file-size-be-optimized"
---
The inherent structure of a PDF, a format designed for fidelity across platforms, often leads to larger file sizes than necessary. Based on my experience working with document management systems, effective optimization isn’t a singular process but a multi-faceted approach focused on the underlying elements of a PDF. I've found significant size reductions are achievable by directly addressing image compression, font embedding, and unnecessary object data, often requiring a combination of techniques.

First, consider image compression. PDFs frequently contain raster images, and the manner in which these images are stored directly impacts overall size. By default, many applications may use lossless compression like PNG, which, while preserving quality, can be substantially larger than lossy options, specifically JPEG. When creating PDFs, evaluating the nature of included images is vital. If a high degree of pixel accuracy isn't critical – such as in photos or colored graphics – JPEG compression at a suitable quality level is often the most effective method to dramatically reduce the size impact of images. Additionally, images should be scaled down to their necessary display dimensions within the PDF. Embedding a high-resolution 3000x2000 pixel photograph intended for a 500x300 pixel space is highly inefficient. The extra data simply inflates the PDF with no corresponding visual benefit.

Font embedding also significantly contributes to PDF file size. While embedding fonts ensures consistent display across devices, large or complex font families can add substantial overhead. I've encountered cases where a PDF using multiple variations of a single font family could be reduced by several megabytes by selectively embedding only the required subsets—the specific characters used in the document—rather than the entire font. This tactic is particularly beneficial when the PDF relies heavily on text. However, the decision to embed fonts at all depends on the requirements for document fidelity. If perfect visual consistency is not paramount, using common system fonts could avoid embedding entirely, thus reducing size. Furthermore, some PDF producers might mistakenly embed the same font multiple times, an inefficiency that requires manual inspection to correct.

Finally, the structure of the PDF document itself contributes to file size. Often, I’ve found that applications might add unnecessary or redundant metadata, object streams, and annotations that contribute to the file’s size. Cleaning these up through optimization software is often necessary. For example, older document editing software might save redundant internal metadata. Moreover, when a PDF is created through multiple manipulations from several different sources, it may contain invisible layers or extraneous elements that unnecessarily inflate the file.

Here are examples demonstrating PDF optimization using Python, which I’ve found useful:

```python
# Example 1:  Image compression with Pillow and pikepdf
from PIL import Image
from pikepdf import Pdf, PdfImage

def compress_images_in_pdf(pdf_path, quality=85):
    """Compress JPEG images within a PDF using Pillow and pikepdf.

        Args:
            pdf_path: The path to the input PDF file.
            quality: JPEG compression quality (0-95), higher is better.
    """
    pdf = Pdf.open(pdf_path)
    for page in pdf.pages:
      for image_obj in page.images.values():
        if image_obj.colorspace == "DeviceRGB" or image_obj.colorspace == "DeviceGray":
            try:
                img_data = bytes(image_obj.raw_image)
                image = Image.frombytes(mode=image_obj.colorspace,size=(image_obj.width, image_obj.height), data=img_data)
                
                # Convert image to JPEG in memory
                temp_buffer = io.BytesIO()
                image.convert("RGB").save(temp_buffer, "jpeg", quality=quality)
                
                # Update the image in the PDF
                compressed_data = temp_buffer.getvalue()
                image_obj.replace(PdfImage(compressed_data))
                
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
    pdf.save(pdf_path.replace(".pdf", "_compressed.pdf"))

# Usage
# compress_images_in_pdf("my_large_pdf.pdf", quality=75)
```

This first script uses `pikepdf` to access PDF data and `Pillow` to manipulate image data. It iterates through each image in the PDF, checks if it’s in RGB or grayscale color space, converts the data to an image object using `PIL`, compresses it to JPEG at the specified quality, and then updates the PDF object with the compressed image, saving the updated document as `_compressed.pdf`. The `try-except` block is crucial here because not all data in a PDF might represent a simple image. This script focuses on lossy image compression as a primary optimization strategy. It requires `pikepdf` and `Pillow` to be installed (e.g. `pip install pikepdf pillow`).

```python
# Example 2: Subsetting fonts with pikepdf (Requires fontTools)
from pikepdf import Pdf
from fontTools import subset
from fontTools.ttLib import TTFont
import io

def subset_fonts_in_pdf(pdf_path):
    """Subsets fonts within a PDF document using pikepdf and fontTools.
       Args:
          pdf_path: Path to the input PDF file.
    """
    pdf = Pdf.open(pdf_path)
    for font in pdf.fonts:
        try:
            if font.is_embedded:
              font_data = font.fontfile_data
              # Load the font with fontTools
              font_obj = TTFont(io.BytesIO(font_data))
              
              # Get subset of glyphs used in the PDF
              text_used = font.text_used
              if text_used:
                subsetter = subset.Subsetter()
                subsetter.populate(text_used)
                subsetter.subset(font_obj)
                
                # Save subsetted font back into the PDF
                temp_buffer = io.BytesIO()
                font_obj.save(temp_buffer)
                font.fontfile_data = temp_buffer.getvalue()
        except Exception as e:
            print(f"Error subsetting font: {e}")
            continue
    pdf.save(pdf_path.replace(".pdf", "_subsetted.pdf"))

# Usage:
# subset_fonts_in_pdf("my_large_text_pdf.pdf")
```
This script uses `fontTools`, a dedicated font manipulation library, along with `pikepdf`. It examines each embedded font, reads its raw data, extracts the unique characters used within the PDF, creates a subset of the font containing only those specific characters, and replaces the original embedded font with this smaller subset. This approach directly addresses the size impact of comprehensive font embedding. Error handling via the `try-except` block ensures the application doesn’t stop if one specific font processing goes wrong. It requires both `pikepdf` and `fontTools` to be installed (e.g. `pip install pikepdf fontTools`).

```python
# Example 3: Removing redundant and unreferenced objects with pikepdf.
from pikepdf import Pdf

def remove_unused_objects(pdf_path):
    """Removes unused or unreferenced objects to reduce file size."""

    pdf = Pdf.open(pdf_path)
    pdf.remove_unreferenced_resources()
    pdf.save(pdf_path.replace(".pdf", "_cleaned.pdf"))

# Usage:
# remove_unused_objects("my_large_messy_pdf.pdf")
```

This final code example is much simpler, and utilizes only `pikepdf`. The `remove_unreferenced_resources()` function will remove objects and data from a PDF that are not being used by the displayed pages. This includes orphaned objects from multiple save and edits, metadata, or other non-essential items. It should be executed as a final step, after compression or font subsetting, to further reduce the file size by cleaning up any remaining inefficiencies.

In addition to scripting, dedicated PDF tools are valuable resources. Software packages that focus on professional PDF creation and modification often offer built-in optimization features covering these elements. These tools offer graphical interfaces, making these options more accessible. Research and testing with a selection of PDF optimization strategies is essential to determine the most suitable solution for a specific case. For instance, a PDF with mainly photographs will benefit most from image compression, while a document focused on text benefits the most from font subsetting. It's also worth consulting documentation related to the specific PDF generation software one is using, as often, there are options there to better control output size during the initial generation process.
