---
title: "Can the sec-api module's ExtractorApi handle 10-Q filings?"
date: "2024-12-23"
id: "can-the-sec-api-modules-extractorapi-handle-10-q-filings"
---

Okay, let's tackle this one. I've had my fair share of experience wrangling financial data, specifically those pesky sec filings, and the question about the `sec-api` module's `ExtractorApi` handling 10-Q filings is a good one. It hits on several practical challenges that often surface when automating data extraction. In short, yes, the `ExtractorApi` *can* handle 10-Q filings, but the devil, as they say, is in the details. It's not a simple 'plug and play' scenario, and you'll quickly find yourself needing a strategic approach to extract the precise data you're after.

Let's unpack that. When I first encountered `sec-api`, I thought, like many do, that it would be a silver bullet. I was tasked with building a system to monitor changes in financial statements for a hedge fund client. 10-Qs, of course, were a critical part of that. We initially tried naive extraction methods – basically, just grabbing all text – but that proved to be unusable noise. The problem isn't necessarily the *ability* of `ExtractorApi` to parse the filing; it's about navigating the document structure, identifying key tables, and dealing with the variance in formatting across different filings. 10-Qs are not cookie-cutter documents; the presentation can differ substantially across companies and even from quarter to quarter for the same company.

The core challenge lies in defining precise extraction rules. The `ExtractorApi` itself provides functionality to apply xpath selectors, which is powerful, but *you* have to determine which xpaths to use for the data points you require. This isn’t a magical process. For example, let’s say you want the consolidated balance sheet. You can’t just assume it’s always under a `<h>Consolidated Balance Sheets</h>` header or a specific table id. I recall struggling intensely with this initially. Sometimes it’s nested within div tags, sometimes there are different captions or titles to tables, and the class names they use are incredibly inconsistent. You might even find that a specific table that was readily available in previous filings is suddenly formatted differently in the next one. Therefore, a static, hardcoded xpath won't survive long. We need to build more robust extraction methods.

To illustrate this, let’s explore a few examples with Python and a conceptual `sec-api` usage. Assume we have a theoretical `ExtractorApi` object as `extractor` (this assumes that the `sec-api` library is correctly installed and configured, which is outside the scope of this response but crucial in real-world scenarios). Here's the first:

```python
# Example 1: Basic extraction using a specific xpath (naive approach)

try:
    filing_document = extractor.get_document_from_filing('accession_number_of_10q')
    # This XPath is highly specific and likely to break in different filings
    revenue_xpath = "//table[@summary='Consolidated Income Statements']/tbody/tr[2]/td[2]/text()"
    revenue_data = filing_document.xpath(revenue_xpath)

    if revenue_data:
        print(f"Extracted revenue: {revenue_data}")
    else:
        print("Revenue data not found using this xpath.")
except Exception as e:
    print(f"Error during extraction: {e}")

```

This code demonstrates the simple use of an xpath, but it’s fragile. If the table summary changes, if rows shift, or if a class changes, the extraction breaks.

Let’s move on to a better method, demonstrating the need for more contextual awareness. This second example focuses on finding a table based on the text near the table header. This strategy utilizes text searching which is slightly less brittle to structural changes than hardcoded row indexes.

```python
# Example 2: Extraction based on table header text

try:
    filing_document = extractor.get_document_from_filing('another_accession_number_of_10q')
    target_header = "Consolidated Balance Sheets"
    table_xpath = f"//table[contains(., '{target_header}')]"
    balance_sheet_table = filing_document.xpath(table_xpath)

    if balance_sheet_table:
        # You would further process the table here to extract relevant rows/columns.
        # For simplicity, we'll just print the first row.
        first_row = balance_sheet_table[0].xpath(".//tr[1]/td/text()")
        print(f"Balance sheet first row: {first_row}")
    else:
        print(f"Table with header '{target_header}' not found.")

except Exception as e:
    print(f"Error during extraction: {e}")
```

This second code example is somewhat better. Instead of relying on specific table ids, we search for a table containing a text string associated with the table header. This makes it more resilient to changes in formatting, but it’s still not perfect. For example, consider scenarios where the header may contain slight variations like "Condensed Consolidated Balance Sheets".

Here's the third, more advanced, example. This example illustrates an iterative search approach, incorporating multiple heuristics for selecting the table of interest:

```python
# Example 3: Robust iterative table search

def find_balance_sheet(filing_document):
    potential_headers = ["Consolidated Balance Sheets", "Condensed Consolidated Balance Sheets", "Balance Sheets"]
    for header in potential_headers:
        table_xpath = f"//table[contains(., '{header}')]"
        tables = filing_document.xpath(table_xpath)
        if tables:
            return tables[0]  # Return the first found table

    return None

try:
    filing_document = extractor.get_document_from_filing('yet_another_accession_number_of_10q')
    balance_sheet_table = find_balance_sheet(filing_document)

    if balance_sheet_table:
        # Further processing...
        print("Balance sheet table found and selected successfully.")
        # example processing : Extract the first few rows
        for row in balance_sheet_table.xpath(".//tr[position() <= 3]"):
            row_data = [td.text for td in row.xpath(".//td")]
            print(row_data)
    else:
         print("Balance sheet not found using any identified header strings")

except Exception as e:
    print(f"Error: {e}")
```

This third example showcases a more production ready approach. It's far more resilient to variations in headers by trying multiple different headers as possible search criteria to discover the relevant table. It emphasizes that robust extraction needs iterative logic that considers multiple conditions, not just a single xpath selector. The table is processed after it has been located successfully in this case to print out the first three rows.

To truly master `sec-api` and 10-Q filings, I recommend studying techniques used for information retrieval and document understanding. In particular, explore resources on:
*   **XPath and XSLT:** Learn these thoroughly to accurately target elements in the filings. Michael Kay’s *XSLT 2.0 and XPath 2.0* is excellent.
*   **Document Object Model (DOM) traversal and manipulation:** Understanding the DOM helps you see how these documents are actually structured and allows you to make well-informed decisions about data extraction. Consider W3C documentation.
*   **Natural Language Processing (NLP):** If the document formats become too varied, or you need to extract text-based insights, NLP techniques can assist in understanding and parsing content. Consult *Speech and Language Processing* by Dan Jurafsky and James H. Martin.
*   **Machine learning for table detection and recognition:** This is an advanced technique for table extraction but crucial to build robust and truly scalable extraction services. A survey article on table detection will be a good starting point here.

In summary, `sec-api`’s `ExtractorApi` can handle 10-Q filings, but the extraction process is far from trivial. It requires a thoughtful approach, careful xpath construction, and the understanding that filings will always present some level of variation. The key is building a flexible extraction strategy that is not dependent on brittle rules. You'll need to continuously refine and adapt your methods as new filings and formats emerge. It’s not about finding a quick fix, but understanding the underlying complexities and creating robust, intelligent extraction pipelines.
