---
title: "How can Cloud Document AI be used to extract and return key-value pairs from documents?"
date: "2025-01-30"
id: "how-can-cloud-document-ai-be-used-to"
---
Cloud Document AI, specifically its specialized processors, offers a powerful means for extracting key-value pairs from unstructured or semi-structured documents. My past projects involved integrating it with several client systems for automated data entry, and the experience revealed its reliance on a combination of pre-trained and custom models to achieve accuracy. The core functionality hinges on the ability of these models to identify semantic relationships between text elements, differentiating, for instance, a customer's name from their address on an invoice.

The fundamental process involves first, feeding a document to a Document AI processor. This could be a PDF, image, or even a scanned document. These processors, offered via the Google Cloud Platform, have different specializations. Some are specifically trained for forms, others for invoices, and some for general purpose document processing. The processor analyzes the document, employing optical character recognition (OCR) to convert any image or PDF into machine-readable text, and then using natural language processing (NLP) techniques to understand the context and semantics of this text. This analysis yields an output document, usually in JSON format. This JSON response details the location of each element detected, their content, and any additional information like confidence scores.  Crucially, the JSON structure includes detected key-value pairs if the processor is trained or configured to extract them.

The key-value pairs are not explicitly marked in the original document. The Document AI processor extracts them based on patterns recognized during its training phase. For instance, a processor trained on invoices may identify any text followed by a colon, a colon followed by a numerical value, or a line above a block of text, as potential key-value pairs. The processor will also use the context of surrounding text (e.g. “Invoice Number:” followed by digits) to identify and extract the correct value. The effectiveness of this process significantly depends on the training data, the quality of the input documents, and the specific processor’s strengths. While many pre-trained processors offer strong performance for common document types, creating a custom processor is critical when dealing with unique document formats. This involves supplying labelled examples to Document AI, allowing the model to understand unique layouts and specific key-value relationships that pre-built models might miss.

My experience shows that a generic approach for key-value pair extraction is insufficient.  It often requires a specific configuration and in some cases, post-processing of the extracted output. Furthermore, the level of extraction accuracy also requires periodic review and adjustments, especially when document formats frequently change. Now, consider specific examples to better understand how this works in practice.

**Example 1: Extracting Invoice Data with a Pre-trained Processor**

This example shows how to interact with a pre-trained processor designed for invoices using Python, assuming the Google Cloud Client library is installed and credentials are set up.

```python
from google.cloud import documentai_v1 as documentai

def extract_invoice_data(project_id, location, processor_id, file_path):
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    with open(file_path, "rb") as f:
        raw_document = f.read()

    document = documentai.Document()
    document.content = raw_document
    document.mime_type = "application/pdf"

    request = documentai.ProcessRequest(name=name, document=document)
    result = client.process_document(request=request)
    document = result.document

    for entity in document.entities:
        if entity.type_ == "key_value_pair":
            print(f"Key: {entity.properties[0].normalized_value.text}")
            print(f"Value: {entity.properties[1].normalized_value.text}")
            print("---")


project_id = "your-gcp-project-id" # Replace with your project ID
location = "us"  # Replace with your location
processor_id = "your-invoice-processor-id"  # Replace with your invoice processor ID
file_path = "path/to/your/invoice.pdf"  # Replace with your invoice file path

extract_invoice_data(project_id, location, processor_id, file_path)
```

In this example, the code initializes a Document AI client, specifies the project, location, and the *pre-trained* invoice processor ID. It then reads the document content and defines the mime type for processing. Finally, it sends the document to the processor and then iterates through each returned entity.  The `if entity.type_ == "key_value_pair"` statement filters for the correct entries, and then it extracts and prints the key and value using the normalized values. The normalization step accounts for OCR inaccuracies and provides a cleaner representation of the values. The key assumption here is the invoice format is common enough to be recognized correctly by the pre-trained model.

**Example 2: Using a Custom Processor with Labeled Data**

In this scenario, we have a unique form for which the pre-trained processor fails to correctly identify the key-value pairs. Therefore, a custom processor must be created and trained using sample documents labeled to indicate the desired extraction points. I will not include the training process in this example, because that's outside the scope of this single program, but will demonstrate retrieving the custom processor for use.

```python
from google.cloud import documentai_v1 as documentai

def extract_form_data(project_id, location, processor_id, file_path):
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    with open(file_path, "rb") as f:
        raw_document = f.read()

    document = documentai.Document()
    document.content = raw_document
    document.mime_type = "image/jpeg"

    request = documentai.ProcessRequest(name=name, document=document)
    result = client.process_document(request=request)
    document = result.document

    for page in document.pages:
        for annotation in page.annotations:
           if annotation.type_ == "key_value_pair":
            key =  document.text[annotation.segments[0].start_index:annotation.segments[0].end_index]
            value = document.text[annotation.segments[1].start_index:annotation.segments[1].end_index]

            print(f"Key: {key.strip()}")
            print(f"Value: {value.strip()}")
            print("---")


project_id = "your-gcp-project-id" # Replace with your project ID
location = "us"  # Replace with your location
processor_id = "your-custom-processor-id"  # Replace with your custom processor ID
file_path = "path/to/your/form.jpg"  # Replace with your form image path

extract_form_data(project_id, location, processor_id, file_path)
```
This example shares a similar structure with the first example but makes use of a *custom* processor, which has been pre-trained on specific forms.  The crucial difference lies in the `processor_id` variable, which is set to refer to our custom-trained model. Additionally, the key and value are extracted directly from the document text, using start and end indexes derived from the annotation segments, ensuring a closer association with the original layout.  The `.strip()` method removes any whitespace that might pad the text segment. The code assumes we labeled the custom processor to identify key-value pairs under the generic annotation type `"key_value_pair"`.

**Example 3: Handling Table Structures within Documents**

Sometimes key-value pairs can exist within table structures, which are not traditionally handled by general key-value pair extractors. Document AI has specific processors capable of understanding these table structures. This example demonstrates retrieving data from a table using a processor trained to understand tables.

```python
from google.cloud import documentai_v1 as documentai

def extract_table_data(project_id, location, processor_id, file_path):
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    with open(file_path, "rb") as f:
        raw_document = f.read()

    document = documentai.Document()
    document.content = raw_document
    document.mime_type = "application/pdf"

    request = documentai.ProcessRequest(name=name, document=document)
    result = client.process_document(request=request)
    document = result.document

    for page in document.pages:
        for table in page.tables:
            for row_index, row in enumerate(table.body_rows):
                row_text = [document.text[segment.start_index:segment.end_index] for cell in row.cells for segment in cell.layout.text_segments ]
                if len(row_text) > 0:
                    print(f"Row {row_index+1}: {', '.join(row_text)}")

project_id = "your-gcp-project-id" # Replace with your project ID
location = "us"  # Replace with your location
processor_id = "your-table-processor-id"  # Replace with your table processor ID
file_path = "path/to/your/table_doc.pdf"  # Replace with your pdf document with tables

extract_table_data(project_id, location, processor_id, file_path)
```

This example employs a document processor geared for handling tables. After processing the document, the code iterates over the detected tables, and within each table iterates over the rows. The text from each cell in a row is extracted, and then these are printed to the console as a comma separated list.  This approach avoids assuming that key-value pairs are explicitly marked in text, instead directly pulling rows and columns from a structured table in the input document. This table data might need further downstream processing based on your requirements.

For deeper understanding and further implementation, I recommend reviewing the official Google Cloud Document AI documentation. Pay special attention to resources on pre-trained processor capabilities, custom processor training, document labeling techniques, and managing API credentials. Investigating the documentation on result formats, specifically the JSON response structure, is also crucial for robust data extraction. Furthermore, explore white papers and use case studies around using Document AI for specific business requirements.  Hands-on practice with various documents and processor types, including custom training, will provide valuable experience.
