---
title: "How to separate records in a plain text file (.txt)?"
date: "2025-01-30"
id: "how-to-separate-records-in-a-plain-text"
---
When processing large datasets from legacy systems, I often encounter data dumps in plain text files where records lack consistent delimiters, posing a challenge for structured parsing. Specifically, the absence of standard separators, such as commas or tabs, necessitates a more nuanced approach to record separation. The problem isn't just about the *presence* of a delimiter; it's often about *identifying* a pattern that can be reliably used to reconstruct logical records.

The primary technique I've found effective involves identifying invariant markers or characteristics within the text that indicate the start or end of a record. These markers aren’t always explicit, requiring a deep inspection of the data. For instance, a record might always begin with a specific prefix, end with a consistent suffix, or be of a fixed length. The challenge is codifying this implicit structure into a programmatic logic that can consistently extract each record. This often involves using regular expressions, string manipulations, or a combination of both, contingent on the observed data format.

Consider, for example, a scenario where records in a text file consistently start with the string "RECORD:" followed by a unique identifier and end with a line containing only "ENDRECORD". To extract these, I would first read the file line by line, accumulating text until the "ENDRECORD" marker is found, then process the constructed record and reset the accumulator. This logic provides a reliable way to reconstitute each record, even when record length varies, leveraging the known starting and ending markers.

Here's a Python example illustrating this method:

```python
import re

def extract_records_by_marker(filepath):
    records = []
    current_record = ""
    record_start_pattern = re.compile(r"^RECORD:.*")

    with open(filepath, 'r') as file:
        for line in file:
            if record_start_pattern.match(line):
                if current_record:
                    records.append(current_record.strip())
                current_record = line
            elif line.strip() == "ENDRECORD":
                current_record += line
                records.append(current_record.strip())
                current_record = ""
            else:
                current_record += line
    
    #Append the last record if not ended with "ENDRECORD"
    if current_record:
        records.append(current_record.strip())
    return records

# Example usage:
filepath = "data.txt"
# Assume data.txt contains:
# RECORD:1234
# Name: John Doe
# Age: 30
# ENDRECORD
# RECORD:5678
# Name: Jane Smith
# Email: jane.smith@example.com
# ENDRECORD
# RECORD:9012
# Location: New York City
# ... some data without end marker
extracted_records = extract_records_by_marker(filepath)
for record in extracted_records:
    print(record)
```

In this function, the `extract_records_by_marker` iterates through each line of the provided file. It utilizes a regular expression (`^RECORD:.*`) to detect the start of a new record. When found, it appends the accumulated record to a list, if there is one, and begins a new record. Upon encountering “ENDRECORD”, it completes the current record and then resets the accumulator. The last check is added to handle the case where the last record does not have an end marker, which happens frequently in practice when the text file is generated under an unexpected process. Finally, the extracted records are returned. The example file “data.txt” is crafted to demonstrate a variety of record content and format.

Another typical situation is fixed-width records, where every record is of an identical number of characters. This is a common data format, particularly for output from legacy mainframe systems. While seemingly straightforward, subtle inconsistencies in padding or newlines can complicate the process. For this type of data, I generally employ string slicing based on the known character lengths of each field. It is also useful to have a configurable field definition where you can specify the start index and length of a field so that the extraction can be driven by a configuration rather than hard-coded in a processing logic. This strategy facilitates better maintainability.

Below is an example demonstrating fixed-width parsing:

```python
def extract_fixed_width_records(filepath, field_lengths):
    records = []
    with open(filepath, 'r') as file:
        for line in file:
            record = {}
            start = 0
            for field_name, length in field_lengths.items():
                record[field_name] = line[start:start + length].strip()
                start += length
            records.append(record)
    return records
# Example Usage
filepath = "fixed_data.txt"
# Assume fixed_data.txt contains:
# JohnDoe    30  Male  NewYork
# JaneSmith   25  FemaleLondon
# PeterPan    40  Male Neverland

field_lengths = {
    "Name": 10,
    "Age": 4,
    "Gender": 6,
    "Location":10
}
extracted_records = extract_fixed_width_records(filepath, field_lengths)

for record in extracted_records:
    print(record)
```

This `extract_fixed_width_records` function reads a file line by line, applying string slices based on the provided field lengths. The lengths are specified in a dictionary where the keys are the field names. The output is a list of dictionaries, each representing a parsed record. The `strip` function removes any padding spaces at the edges of each field. This approach handles subtle variances in spacing within the fixed width format.

A less common, but encountered situation, is where records are delimited by specific strings that are not necessarily at the start or end of a line, and may even appear within a record’s data itself. In these cases, careful consideration of the data and the delimiters is essential. For these scenarios, regex can be very helpful. However, if regex is not an option, manual stateful parsing might be required. It can be done by scanning the content character-by-character and building up each record based on the markers. This kind of extraction tends to be more error-prone, so care must be exercised.

Here’s an example of how it could be done using a combination of regex and a stateful approach.

```python
import re
def extract_delimited_records(filepath, delimiter):
    records = []
    current_record = ""
    with open(filepath, 'r') as file:
        file_content = file.read()
    
    split_content = re.split(re.escape(delimiter), file_content)
    for rec in split_content:
        if rec:
            records.append(rec.strip())
    
    return records
# Example usage
filepath = "mixed_data.txt"
# Assume mixed_data.txt contains
# Field1: Value1@@@Field2: Value2@@@Field3: Value3; Otherdata1@@@Field1: ValueA@@@Field2: ValueB@@@Field3: ValueC; Otherdata2

delimiter = "@@@"

extracted_records = extract_delimited_records(filepath, delimiter)
for record in extracted_records:
    print(record)

```
In this function, instead of processing line by line, the whole file content is read, and then the content is split by the delimiter. This relies on the `re.split` functionality to find all occurrences of the specified delimiter, handling cases where it might be within a line. The `re.escape` is used to make the delimiter string is treated as literal and avoid the error caused by meta-characters. This approach assumes that the delimiter can be reliably used as a marker for records.

For deeper understanding of textual data processing, I recommend exploring resources on string manipulation techniques available in most programming languages. In particular, mastering regular expressions is invaluable for pattern matching and data extraction. Texts covering data analysis and manipulation provide theoretical and practical insights into data formats and processing techniques. Familiarizing oneself with established data formats and parsing libraries aids in developing robust and efficient extraction pipelines. In addition to these, books focusing on legacy system integration and data migration can provide context on the types of text-based data formats one might encounter.
