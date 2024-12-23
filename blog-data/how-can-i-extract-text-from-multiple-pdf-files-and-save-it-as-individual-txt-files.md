---
title: "How can I extract text from multiple PDF files and save it as individual .txt files?"
date: "2024-12-23"
id: "how-can-i-extract-text-from-multiple-pdf-files-and-save-it-as-individual-txt-files"
---

, let's get into this. I've tackled this very problem numerous times over the years, and it’s a surprisingly common requirement in various data processing pipelines. Extracting text from PDFs programmatically can seem straightforward initially, but the devil, as they say, is in the details. There’s a whole ecosystem of tools and approaches, and the “best” method often depends heavily on the nature of the PDF itself.

The core challenge lies in the structure of PDFs. Unlike plain text files, PDFs are designed for document presentation, containing not just textual content but also layout information, fonts, images, and potentially much more. This complexity means we need libraries designed specifically to interpret this structure and isolate the text we need. Overly simplistic approaches, like treating a PDF as a raw binary stream, will yield gibberish.

My own experience, working on a large-scale document archival project several years back, highlighted this dramatically. We were initially using a naive system that relied heavily on OCR for every single PDF, even those with embedded text layers. The performance was abysmal, and the accuracy was often questionable, particularly with older or low-quality scans. We ended up completely rewriting that module, focusing on smarter text extraction and OCR as a fallback only.

The key is to use a library with strong PDF parsing capabilities. There are several options, each with trade-offs in terms of ease of use, performance, and the range of supported features. In my projects, I've predominantly relied on `PyPDF2` and `pdfminer.six` in python due to their flexibility and community support. For more demanding tasks, `Tika` with its various language bindings proves to be a robust, albeit more complex choice. I will use `PyPDF2` and `pdfminer.six` to illustrate my approach since they're quite popular and offer a good balance of simplicity and power. Let's look at some code examples:

**Example 1: Using `PyPDF2` (Simple Cases)**

`PyPDF2` is a great starting point for relatively simple PDFs. It excels at handling PDFs with clear, embedded text layers. However, it often struggles with scanned documents or PDFs where the text isn't properly encoded.

```python
import os
from PyPDF2 import PdfReader

def extract_text_pypdf2(pdf_path, output_dir):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        
        file_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text)
        print(f"Extracted text from {pdf_path} to {output_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    pdf_directory = "pdf_files" # Assumes you have a directory named pdf_files with .pdf files in it
    output_directory = "txt_files"
    os.makedirs(output_directory, exist_ok=True)

    for file in os.listdir(pdf_directory):
        if file.endswith(".pdf"):
             pdf_path = os.path.join(pdf_directory, file)
             extract_text_pypdf2(pdf_path, output_directory)

```
This script iterates through the PDF files in a specified directory. For each file, it uses `PdfReader` to read each page, extracts the text using the `extract_text()` method, and concatenates it. The resulting text is then saved into a corresponding .txt file in the output directory.

**Example 2: Using `pdfminer.six` (More Complex Cases)**

For more complex PDFs where `PyPDF2` may stumble, `pdfminer.six` is often more reliable. It's lower-level and gives you finer control over the text extraction process. However, it can be a bit more involved to set up.

```python
import os
from pdfminer.high_level import extract_text

def extract_text_pdfminer(pdf_path, output_dir):
    try:
        text = extract_text(pdf_path)
        file_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text)
        print(f"Extracted text from {pdf_path} to {output_path}")

    except Exception as e:
      print(f"Error processing {pdf_path}: {e}")


if __name__ == "__main__":
    pdf_directory = "pdf_files"
    output_directory = "txt_files"
    os.makedirs(output_directory, exist_ok=True)

    for file in os.listdir(pdf_directory):
      if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, file)
        extract_text_pdfminer(pdf_path, output_directory)
```
Here, we utilize the `extract_text` function from `pdfminer.high_level`. This is a much higher-level, easier way to approach `pdfminer.six`. It handles parsing the content of the pdf, and extracts all text with less effort than writing custom logic. This approach is quite effective for a broader range of PDFs.

**Example 3: Error Handling and Improvements**

The previous examples provide the fundamentals, but real-world scenarios often require more robust error handling and refinements. Here's an example incorporating basic error checking and a slight modification in how we determine the file path for more flexible processing:

```python
import os
from PyPDF2 import PdfReader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pdf(pdf_path, output_dir):
    if not pdf_path.lower().endswith('.pdf'):
       logging.warning(f"Skipping non-PDF file: {pdf_path}")
       return

    try:
        with open(pdf_path, 'rb') as file:
             reader = PdfReader(file)
             text = ""
             for page_num in range(len(reader.pages)):
                 page = reader.pages[page_num]
                 text += page.extract_text()
        file_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
        logging.info(f"Extracted text from {pdf_path} to {output_path}")
    except FileNotFoundError:
        logging.error(f"Error: PDF file not found at path: {pdf_path}")
    except Exception as e:
        logging.error(f"Error processing PDF file {pdf_path}: {e}")

if __name__ == "__main__":
    pdf_directory = "pdf_files"
    output_directory = "txt_files"
    os.makedirs(output_directory, exist_ok=True)


    for filename in os.listdir(pdf_directory):
         pdf_path = os.path.join(pdf_directory, filename)
         process_pdf(pdf_path, output_directory)

```

In this revised version, we've introduced logging to record information, warnings, and errors. The pdf extension check ensures that the function will skip any non .pdf files if they happen to reside in the pdf directory. The `try-except` blocks will handle file not found scenarios and general errors gracefully, making the application more robust to real-world variations in file structure.

For deeper dives into the underlying mechanisms, I'd strongly recommend "PDF Explained" by John Whitington. It covers the PDF specification in considerable detail and is invaluable for understanding the intricacies of document structure. Another excellent resource is “Programming with PDF” by Doug Beeferman, it is very practical and explains many of the common issues when working with PDFs programmatically. Also, delving into the official documentation of libraries like `PyPDF2` and `pdfminer.six` is highly beneficial.

In conclusion, while the fundamental task of extracting text from multiple PDFs to individual text files might seem trivial at first glance, practical implementations require careful library selection and robust error handling. These examples represent a solid foundation and should be adaptable to most situations. Keep refining your approach as the nuances of the PDFs themselves require it.
