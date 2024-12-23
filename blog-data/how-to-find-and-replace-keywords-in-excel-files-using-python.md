---
title: "How to find and replace keywords in Excel files using Python?"
date: "2024-12-16"
id: "how-to-find-and-replace-keywords-in-excel-files-using-python"
---

Alright,  I remember a particularly messy project a few years back. We were dealing with hundreds of Excel reports, each containing slightly different variations of product codes and descriptions. Manually editing those was clearly out of the question, so a script was necessary. That's when i really got familiar with the ins and outs of programmatically modifying excel files with python. It's not always as straightforward as just opening a text file and doing a string replace, but the libraries available to us make the task fairly manageable.

The core challenge lies in understanding excel's internal structure. It’s not just plain text; it’s a structured, binary format. This means we can't rely on simple text manipulation methods. Instead, we use specialized libraries that interpret the excel file format and allow us to interact with the data as a set of rows, columns, and cells. For python, the two primary contenders are `openpyxl` and `xlrd`/`xlwt` (or their successor `xlwings`, more powerful but slightly more involved), depending on the format. I usually prefer `openpyxl` for modern `.xlsx` files due to its simplicity and the fact that it natively handles `xlsx` files (which are zip archives containing xml files). `xlrd` was very capable for the old `.xls` format, but is read only now, and its write counterpart `xlwt` is less convenient than openpyxl.

Let's consider the typical scenario: you need to find all instances of a keyword in a spreadsheet and replace it with another. The process involves several key steps: opening the file, iterating over all cells in relevant sheets, identifying cells that contain the keyword, and performing the replacement. Here's how we can approach this, starting with some illustrative code.

```python
import openpyxl

def find_and_replace_excel(filename, old_keyword, new_keyword, sheet_name=None):
    """
    Finds and replaces a keyword in an Excel file.

    Args:
        filename (str): The path to the Excel file.
        old_keyword (str): The keyword to search for.
        new_keyword (str): The keyword to replace with.
        sheet_name (str, optional): The name of the sheet to process. If None, all sheets are processed.

    Returns:
         bool: True if replacements were made, False otherwise.
    """
    wb = openpyxl.load_workbook(filename)
    replaced = False
    if sheet_name:
        if sheet_name not in wb.sheetnames:
            print(f"Sheet '{sheet_name}' not found.")
            return False
        sheets_to_process = [wb[sheet_name]]
    else:
       sheets_to_process = wb.worksheets

    for sheet in sheets_to_process:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None and isinstance(cell.value, str):
                    if old_keyword in cell.value:
                        cell.value = cell.value.replace(old_keyword, new_keyword)
                        replaced = True

    if replaced:
      wb.save(filename)
    return replaced


# Example Usage
if __name__ == "__main__":
    file_path = 'example.xlsx'  # Ensure example.xlsx exists or create a dummy one for testing
    old_word = "old_product"
    new_word = "new_product"
    result = find_and_replace_excel(file_path, old_word, new_word)
    if result:
        print(f"Replaced '{old_word}' with '{new_word}' in the file '{file_path}'.")
    else:
        print(f"'{old_word}' not found in the file '{file_path}'.")
```

This basic example opens the excel file, iterates over all cells in each worksheet, and, if the value of a cell is a string and contains the `old_keyword`, replaces it with the `new_keyword`. A crucial detail is handling the case of a cell value not being a string (e.g., numerical or datetime) which prevents errors. We check for the `None` case and also explicitly use `isinstance(cell.value, str)`. The modified workbook is then saved. This was my go-to solution for the initial problem of straightforward keyword changes.

However, sometimes we have to deal with more intricate cases, where only some cells need to be altered. Suppose you only want to replace the keyword in cells in the first column. This is the kind of problem where things get a bit more nuanced. Here's how we'd adapt the previous code to handle that requirement:

```python
import openpyxl

def find_and_replace_first_column(filename, old_keyword, new_keyword, sheet_name=None):
    """
    Finds and replaces a keyword in the first column of an Excel file.

    Args:
        filename (str): The path to the Excel file.
        old_keyword (str): The keyword to search for.
        new_keyword (str): The keyword to replace with.
        sheet_name (str, optional): The name of the sheet to process. If None, all sheets are processed.

    Returns:
        bool: True if replacements were made, False otherwise.
    """
    wb = openpyxl.load_workbook(filename)
    replaced = False

    if sheet_name:
       if sheet_name not in wb.sheetnames:
            print(f"Sheet '{sheet_name}' not found.")
            return False
       sheets_to_process = [wb[sheet_name]]
    else:
       sheets_to_process = wb.worksheets

    for sheet in sheets_to_process:
        for row in sheet.iter_rows():
           cell = row[0]  # Access the first cell of each row
           if cell.value is not None and isinstance(cell.value, str):
              if old_keyword in cell.value:
                cell.value = cell.value.replace(old_keyword, new_keyword)
                replaced = True

    if replaced:
        wb.save(filename)
    return replaced


# Example Usage
if __name__ == "__main__":
    file_path = 'example.xlsx' # Ensure example.xlsx exists or create a dummy one for testing
    old_word = "old_category"
    new_word = "new_category"
    result = find_and_replace_first_column(file_path, old_word, new_word)
    if result:
        print(f"Replaced '{old_word}' with '{new_word}' in the first column of '{file_path}'.")
    else:
       print(f"'{old_word}' not found in the first column of '{file_path}'.")

```

This code focuses on the first cell in each row (`cell = row[0]`). We access the cells by index within each row instead of using nested loops, which is a more targeted approach. This is critical when working with complex spreadsheets where only very specific cells need to be modified. It prevents accidental replacements in other columns.

Now, a final common scenario i dealt with involved not just finding simple string instances, but making replacements based on patterns (regular expressions). For instance, you might want to replace all occurrences of "product-001" to "product-002" in a more flexible way allowing for potentially any number instead of '001'. Here is how we can do that:

```python
import openpyxl
import re

def find_and_replace_regex(filename, regex_pattern, new_value, sheet_name=None):
    """
    Finds and replaces a pattern (regex) in an Excel file.

    Args:
        filename (str): The path to the Excel file.
        regex_pattern (str): The regular expression pattern to search for.
        new_value (str): The value to replace with.
        sheet_name (str, optional): The name of the sheet to process. If None, all sheets are processed.

    Returns:
        bool: True if replacements were made, False otherwise.
    """
    wb = openpyxl.load_workbook(filename)
    replaced = False
    if sheet_name:
        if sheet_name not in wb.sheetnames:
            print(f"Sheet '{sheet_name}' not found.")
            return False
        sheets_to_process = [wb[sheet_name]]
    else:
        sheets_to_process = wb.worksheets

    for sheet in sheets_to_process:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None and isinstance(cell.value, str):
                    if re.search(regex_pattern, cell.value):
                        cell.value = re.sub(regex_pattern, new_value, cell.value)
                        replaced = True

    if replaced:
       wb.save(filename)
    return replaced


# Example Usage
if __name__ == "__main__":
    file_path = 'example.xlsx'  # Ensure example.xlsx exists or create a dummy one for testing
    regex_pattern = r"product-\d{3}" # Replace product- followed by 3 digits
    new_val = "new_product_code"
    result = find_and_replace_regex(file_path, regex_pattern, new_val)
    if result:
      print(f"Replaced pattern '{regex_pattern}' with '{new_val}' in the file '{file_path}'.")
    else:
        print(f"Pattern '{regex_pattern}' not found in the file '{file_path}'.")
```

Here, we’ve integrated the `re` module for regular expressions. The `re.search()` method checks for a pattern match, and `re.sub()` method does the replacement. This gives you a lot more power when you need to handle complex variations in cell contents. Using `r"product-\d{3}"` for example, we can match `product-001`, `product-042`, and so on.

For a deeper dive into the nitty-gritty of excel file formats, I recommend exploring the ECMA-376 standard for office Open XML file formats. Specifically, part 1 details the package conventions (which helps understand the zip archive structure), and parts 2-4 cover the XML schemas used in excel files. While this is a dense read, it provides a very thorough understanding. For practical usage and examples, the official documentation of `openpyxl` is invaluable. These references will definitely deepen your understanding beyond simple examples provided here. I hope this breakdown of approaches helps you with your particular problem!
