---
title: "How do I find and replace keywords in Excel files using Python?"
date: "2024-12-23"
id: "how-do-i-find-and-replace-keywords-in-excel-files-using-python"
---

Let’s tackle this head-on. Replacing keywords in excel files using python is a common task, and it's something I've personally dealt with quite a few times, especially back in my days managing large datasets for financial reporting. It’s not overly complicated, but getting it efficient and robust requires a little more than just slapping together a script.

The core of the issue hinges on understanding how excel files are structured and how python can interact with that structure. Essentially, we're dealing with a spreadsheet, which is a collection of rows and columns, where each cell can hold a different type of data, including strings (which are where our keywords reside). Python doesn’t directly operate on the binary structure of .xlsx files. Instead, we use libraries like `openpyxl` or `xlrd` and `xlwt` (for older .xls files). These libraries offer an abstraction layer, allowing us to work with excel data as python objects, typically as dictionaries or lists of lists, making it much easier to find and replace text. `Openpyxl` is particularly useful for .xlsx files, which is the modern standard, and it's the one I'd recommend you use in most cases.

The basic process involves three main steps. First, you load the excel file into python. Second, you iterate through each cell in the sheet, checking if the cell contains any of the keywords. If it does, then you replace it with the new text. Finally, after making all the necessary replacements, you save the modified spreadsheet. Simple enough when stated like that, but naturally, details matter, especially when you're dealing with very large files. Let's delve into a few examples.

**Example 1: Basic Keyword Replacement in a Single Sheet**

This example focuses on a single worksheet, assuming the keywords are simple strings, not regular expressions. We’ll iterate through every cell and apply the replacement directly.

```python
import openpyxl

def replace_keywords_basic(excel_file, keywords, replacement):
    """Replaces all occurrences of specified keywords with the replacement string in a given excel file."""
    workbook = openpyxl.load_workbook(excel_file)
    sheet = workbook.active # get the active sheet

    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is not None and isinstance(cell.value, str): # ensure we're working with text only.
                for keyword in keywords:
                    if keyword in cell.value:
                        cell.value = cell.value.replace(keyword, replacement)

    workbook.save(excel_file)
    print(f"Keywords replaced in {excel_file} successfully.")

# example of calling
if __name__ == '__main__':
    file_path = "sample_excel.xlsx" # Replace with path of your excel file.
    keywords_to_replace = ["oldTerm1", "oldTerm2", "oldTerm3"]
    replacement_term = "newTerm"
    replace_keywords_basic(file_path, keywords_to_replace, replacement_term)
```

Here, we load the workbook using `openpyxl.load_workbook()`. We then iterate through each cell of the active sheet. We use `cell.value is not None and isinstance(cell.value, str)` to ensure that we are only comparing strings and avoid unexpected errors if there are integers, dates, or blank cells. We then check if any of our `keywords` are present, and if so, we do a direct `replace()`. Finally, we save the updated workbook. This approach is fine for smaller files, but can become inefficient for large excel files with many keywords.

**Example 2: Handling Multiple Sheets and Partial String Matching with Regular Expressions**

This example enhances the previous one, adding the ability to handle multiple sheets and use regular expressions for more complex matching. Regular expressions can help handle partial words, or a variety of spelling issues for keywords, for example.

```python
import openpyxl
import re

def replace_keywords_regex(excel_file, keywords_dict, case_sensitive=False):
    """Replaces keywords in multiple sheets using regular expressions."""
    workbook = openpyxl.load_workbook(excel_file)
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None and isinstance(cell.value, str):
                    for keyword, replacement in keywords_dict.items():
                        flags = 0 if case_sensitive else re.IGNORECASE # handling case sensitivity for regex
                        cell.value = re.sub(keyword, replacement, cell.value, flags=flags)

    workbook.save(excel_file)
    print(f"Keywords replaced using regular expressions in {excel_file} successfully.")

if __name__ == '__main__':
    file_path = "sample_excel_regex.xlsx"
    keywords_to_replace_dict = {
        r'\b(old)\s+(Term1)\b': "newTermA", #whole word matching using word boundaries '\b'
        r'term2[a-z]': "newTermB",  #matching any term2 with following characters
        r'old term3\d+': "newTermC", #matching old term3 followed by a digit.
    }
    replace_keywords_regex(file_path, keywords_to_replace_dict, case_sensitive=False)
```

Here, we use `workbook.sheetnames` to iterate through all sheets in the workbook. We use a dictionary `keywords_dict` to map keywords to their replacements. We use `re.sub()` to perform the replacement based on the provided regular expressions. Note the flags parameter to the regex, this is used to implement an optional case-insensitive replacement. We also use regular expression features like word boundaries and character matching. This is significantly more flexible than simple string matching. This allows for very precise replacements to occur.

**Example 3: Handling Large Files with Chunked Reading (less memory intensive)**

For extremely large excel files, loading the entire sheet into memory can be problematic. To alleviate this, we can use a chunked reading approach. Although openpyxl doesn't provide chunked reading directly, we can simulate this by iterating over chunks of rows. In this example, I'm using an arbitrarily defined chunk size. Depending on your hardware and file size, you will want to adjust this. You’ll need to decide what's optimal for your situation.

```python
import openpyxl
import re

def replace_keywords_chunked(excel_file, keywords_dict, case_sensitive=False, chunk_size=1000):
    """Replaces keywords in excel files using a chunked approach."""
    workbook = openpyxl.load_workbook(excel_file)
    for sheet_name in workbook.sheetnames:
      sheet = workbook[sheet_name]
      max_row = sheet.max_row
      for i in range(1, max_row + 1, chunk_size):
        rows_chunk = sheet.iter_rows(min_row=i, max_row=min(i + chunk_size - 1, max_row))
        for row in rows_chunk:
            for cell in row:
                if cell.value is not None and isinstance(cell.value, str):
                    for keyword, replacement in keywords_dict.items():
                        flags = 0 if case_sensitive else re.IGNORECASE # handling case sensitivity for regex
                        cell.value = re.sub(keyword, replacement, cell.value, flags=flags)

    workbook.save(excel_file)
    print(f"Keywords replaced chunked in {excel_file} successfully.")


if __name__ == '__main__':
    file_path = "large_excel.xlsx"
    keywords_to_replace_dict = {
    r'\b(old)\s+(Term1)\b': "newTermA",
    r'term2[a-z]': "newTermB",
    r'old term3\d+': "newTermC",
    }

    replace_keywords_chunked(file_path, keywords_to_replace_dict, case_sensitive=False)
```

In this version, instead of loading all rows at once, we iterate through rows in chunks, processing a smaller number of rows at a time. This avoids loading large files entirely into memory, which is useful for very large spreadsheets. The overall logic is very similar to the second example, but with the addition of chunking.

**Recommendations for Further Learning**

For more in-depth information, I recommend looking at these authoritative sources:

1.  **"Automate the Boring Stuff with Python" by Al Sweigart:** Provides a very hands-on introduction to various automation tasks, including excel manipulation. Good for getting started and understanding the basics.
2.  **The official documentation for `openpyxl`:** This is the most detailed source of information on all functionalities of the library. When you need the nitty-gritty details, the documentation will have you covered.
3. **"Mastering Regular Expressions" by Jeffrey Friedl**: This classic book is a must-read if you want to fully understand and harness the power of regular expressions, an essential tool for more complex text manipulation.
4. **"Python Cookbook" by David Beazley and Brian K. Jones**: Although not focused solely on Excel files, this book covers various python programming patterns and techniques, including efficient file handling, which can enhance your skills related to this specific task.

In summary, finding and replacing keywords in excel with python is a solvable problem by leveraging powerful libraries like `openpyxl` and regular expression capabilities of `re` library. You can start with simple replacements and graduate towards more complex scenarios such as regex matching and chunked reading when performance and memory management are key considerations.
