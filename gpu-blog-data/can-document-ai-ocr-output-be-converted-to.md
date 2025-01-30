---
title: "Can Document AI OCR output be converted to PDF?"
date: "2025-01-30"
id: "can-document-ai-ocr-output-be-converted-to"
---
Document AI's Optical Character Recognition (OCR) output, typically provided as structured JSON data, can indeed be transformed into a PDF document. This process isn't a direct conversion; rather, it involves re-rendering the extracted text and, potentially, the original image within a PDF container. I’ve encountered this need repeatedly in projects involving archival digitization and automated document processing, where preserving the original layout alongside searchable text is crucial.

The core challenge lies in translating the bounding box coordinates, recognized text, and confidence scores found in the Document AI JSON output into the appropriate PDF instructions. The PDF format, while seemingly straightforward to end users, is internally complex, utilizing a combination of textual elements, raster images, vector graphics, and special rendering instructions. Therefore, a conversion pipeline usually involves several key steps: parsing the Document AI JSON output, structuring the textual data to match the original document layout, and finally, constructing the PDF file using a suitable library.

The conversion isn’t merely about pasting the extracted text into a PDF. Maintaining the positional integrity of the text within the document is paramount. For example, if a document contains a table, the text must be placed into the PDF in such a way as to visually recreate that table structure. Ignoring the bounding boxes or failing to correctly calculate relative positions will result in misaligned, unreadable text. Additionally, we must decide whether to include the original image as a background layer in the PDF. While this preserves the "original look," it increases file size and may reduce text searchability depending on PDF render settings. Sometimes, a 'text-only' PDF (where only the extracted text is rendered over a blank background) is preferable, depending on usage.

Furthermore, the confidence scores returned by Document AI inform how accurately the text was recognized. This can be useful, not directly for conversion to a PDF, but during a later stage for human review and correction. For critical documents, regions with low confidence scores might be flagged for manual inspection. The conversion script should therefore provide information about extracted text along with its certainty level, thus assisting the post-conversion QA.

Let’s examine code examples using Python, a language commonly employed for this type of task, employing the `reportlab` library which is particularly useful for creating PDFs:

**Example 1: Basic Text Placement**

This initial example focuses on the essential mechanics of placing extracted text into a PDF, ignoring complex layouts for now. It assumes a simplified Document AI output where text lines are described by bounding boxes and the recognized string.

```python
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_basic_pdf(doc_ai_data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    for page in doc_ai_data:
        for text_data in page: #Simplified text entry
            x = text_data['bounding_box']['x'] * inch #Assuming unit conversions
            y = text_data['bounding_box']['y'] * inch
            text = text_data['text']
            c.drawString(x, y, text) # Draw text at position
        c.showPage() # New page for every item
    c.save()

# Example JSON input (Simplified)
example_data = [
    [{
        'bounding_box':{'x':1,'y':10},
        'text':'First line of text.'
    }, {
        'bounding_box':{'x':1,'y':9.5},
        'text':'Second line.'
    }],
    [{
        'bounding_box':{'x':1,'y':10},
        'text':'Another page here.'
    }]
]

create_basic_pdf(example_data, "basic_output.pdf")
```

**Commentary:** This simple example establishes the core workflow. It iterates through the simplified document AI data (assuming one array per page). For every ‘text’ item, it extracts the (x, y) coordinates and renders the corresponding extracted string. The crucial part is the mapping of bounding box coordinates to PDF coordinates. Units (like inches) may have to be factored in, depending on what unit Document AI uses. While basic, this demonstrates fundamental text placement with no layout concerns.

**Example 2:  Including Original Image (Overlay)**

This example expands on the first, adding an image overlay to the PDF for increased fidelity to the original document layout.  It assumes that each Document AI output entry has an associated page image path.

```python
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


def create_overlay_pdf(doc_ai_data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    for page in doc_ai_data:
        image_path = page['image_path'] #Assume image_path for page
        img = ImageReader(image_path)
        img_width, img_height = img.getSize()
        c.drawImage(img, 0, 0, width=img_width, height=img_height)
        for text_data in page['text_items']: #Now, nested under text_items
            x = text_data['bounding_box']['x'] * inch
            y = text_data['bounding_box']['y'] * inch
            text = text_data['text']
            c.drawString(x, y, text)
        c.showPage()
    c.save()

# Example JSON input (Expanded)
example_data = [
    {
        'image_path':'./image1.jpg',
         'text_items':[{
            'bounding_box':{'x':1,'y':10},
            'text':'First line of text.'
        }, {
            'bounding_box':{'x':1,'y':9.5},
            'text':'Second line.'
        }]
    },
    {
        'image_path':'./image2.jpg',
        'text_items':[{
            'bounding_box':{'x':1,'y':10},
            'text':'Another page here.'
        }]
    }
]
create_overlay_pdf(example_data, "overlay_output.pdf")
```

**Commentary:** This version takes page images from the `doc_ai_data`. Each image is loaded and rendered as a background using the `drawImage` command. Text is subsequently rendered on top of the image, creating a visual overlay that helps preserve the appearance of the original document.  The data structure is more realistic, nesting text items under each page object. Note, in real applications, appropriate error handling should exist to deal with missing images and other anomalies. `PIL` (Pillow) is another viable library for image manipulation alongside ReportLab.

**Example 3:  Confidence Score Indication**

Building on the previous example, we'll incorporate confidence score indications in the generated PDF using text coloring. This requires some basic conditional logic during the text rendering.

```python
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors


def create_confidence_pdf(doc_ai_data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    for page in doc_ai_data:
        image_path = page['image_path']
        img = ImageReader(image_path)
        img_width, img_height = img.getSize()
        c.drawImage(img, 0, 0, width=img_width, height=img_height)
        for text_data in page['text_items']:
            x = text_data['bounding_box']['x'] * inch
            y = text_data['bounding_box']['y'] * inch
            text = text_data['text']
            confidence = text_data.get('confidence', 1.0)  #Assume default confidence
            if confidence < 0.8: #Colorize text for lower confidence
                c.setFillColor(colors.red)
            else:
                c.setFillColor(colors.black) #Default color
            c.drawString(x, y, text)
            c.setFillColor(colors.black) #Reset color after each draw
        c.showPage()
    c.save()
    
example_data = [
    {
        'image_path':'./image1.jpg',
         'text_items':[{
            'bounding_box':{'x':1,'y':10},
            'text':'First line of text.',
            'confidence':0.95
        }, {
            'bounding_box':{'x':1,'y':9.5},
            'text':'Second line with potential error.',
            'confidence':0.75
        }]
    },
    {
        'image_path':'./image2.jpg',
        'text_items':[{
            'bounding_box':{'x':1,'y':10},
            'text':'Another page here.',
            'confidence':1.0
        }]
    }
]

create_confidence_pdf(example_data, "confidence_output.pdf")
```

**Commentary:**  Here, we access the ‘confidence’ score associated with each text block. If this score is below a threshold (0.8 in this example), the rendered text is displayed in red, potentially flagging it for further inspection. Color coding assists in quality control, allowing users to quickly identify potentially inaccurate text areas in the PDF.

In terms of resources, the `reportlab` user guide is comprehensive. For image handling, documentation for `Pillow (PIL)` is very useful. Additionally, working with the PDF specification itself, although dense, can deepen the understanding of the underlying mechanics. Lastly, many online forums and communities discuss OCR and document processing techniques and provide valuable real-world examples and solutions. When dealing with complex documents like those containing multiple columns, tables and specific formatting, you will likely encounter complexities that extend beyond the basic functionalities shown here; it often requires a combination of sophisticated algorithms, text line grouping, and table reconstruction techniques that have to be tailored to specific cases.
