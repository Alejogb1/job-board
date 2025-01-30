---
title: "How can I pass external variable values to a COBOL module and write them to a new output file?"
date: "2025-01-30"
id: "how-can-i-pass-external-variable-values-to"
---
In my years maintaining legacy systems, I've frequently encountered the challenge of integrating modern data flows with COBOL's procedural nature. Directly passing variables from external systems into a COBOL module and then writing them to a new output file requires understanding COBOL's data handling mechanisms, particularly `LINKAGE SECTION` parameters and file I/O. Here’s how I typically approach this.

The core concept revolves around treating the COBOL module as a subroutine or function called from a higher-level program or a script. This calling program is responsible for preparing and passing the data, while the COBOL module focuses solely on processing that received data and performing its specific task – in this case, writing to a new file.

**Explanation**

When a COBOL module needs to receive external values, these values are not implicitly available; they must be explicitly passed as arguments. We achieve this through the `LINKAGE SECTION`, a special section within a COBOL program where data structures are defined that are not allocated memory internally. Instead, this memory is provided by the calling program.

The `PROCEDURE DIVISION` header will specify the parameters being accepted by the module, matching the data structure described within the `LINKAGE SECTION`. This matching is crucial; type mismatches will likely lead to unpredictable program behavior or runtime errors. These parameters represent a kind of "interface" through which the external values are sent into the COBOL routine.

Within the `PROCEDURE DIVISION`, you'll handle the incoming data. Typically, the data is moved from these linkage variables into working storage variables (declared in the `WORKING-STORAGE SECTION`). The process involves using `MOVE` statements, often followed by further data manipulation, validation, or conditional processing.

The process for writing to a new output file requires the following steps:

1.  **File Definition:** Within the `ENVIRONMENT DIVISION`, you must define the file using `FILE-CONTROL` paragraph. This involves assigning an external file name and a logical file name which the program will use internally.
2.  **File Opening:** Within the `PROCEDURE DIVISION` you must open the file using the `OPEN OUTPUT` statement. This creates the physical output file (if it doesn’t exist) and prepares it for writing operations.
3.  **Data Writing:** Inside your processing logic, after having obtained the necessary data and performed any transformations, use the `WRITE` statement to transfer data to the opened output file.
4.  **File Closing:** After all data has been written, close the file using the `CLOSE` statement. This ensures that all data is properly written to the physical file, preventing potential data loss or corruption.

It's critically important that the record structure used when writing the data matches the structure defined in the `FILE SECTION`. In my experience, mismatches here are common culprits for data corruption issues and unexpected file behavior.

**Code Examples**

Here are three code examples illustrating how different types of data might be passed to a COBOL module and written to an output file. Each example highlights a slightly different scenario and will be annotated with specific comments.

**Example 1: Simple String and Numeric Value**

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. WRITE-SIMPLE-DATA.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT OUT-FILE ASSIGN TO "output.txt"
           ORGANIZATION IS LINE SEQUENTIAL.
       DATA DIVISION.
       FILE SECTION.
       FD  OUT-FILE.
       01  OUT-REC  PIC X(80).
       WORKING-STORAGE SECTION.
       01  WS-NAME PIC X(30).
       01  WS-AGE  PIC 9(03).
       01  WS-OUTPUT-LINE PIC X(80).
       LINKAGE SECTION.
       01  LK-NAME  PIC X(30).
       01  LK-AGE   PIC 9(03).
       PROCEDURE DIVISION USING LK-NAME LK-AGE.
           MOVE LK-NAME TO WS-NAME.
           MOVE LK-AGE  TO WS-AGE.

           OPEN OUTPUT OUT-FILE.
           STRING "Name: " WS-NAME ", Age: " WS-AGE
               DELIMITED BY SIZE INTO WS-OUTPUT-LINE.
           WRITE OUT-REC FROM WS-OUTPUT-LINE.
           CLOSE OUT-FILE.
           GOBACK.
```
*   **Commentary:** This example shows passing a string (`LK-NAME`) and a numeric value (`LK-AGE`) to the module via the `LINKAGE SECTION`. It opens a line sequential output file "output.txt" and formats the output into a string for writing. It demonstrates basic data movement and formatting before writing to the output file. The `STRING` statement constructs a single line of output.

**Example 2: Structured Record Data**

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. WRITE-STRUCTURED-DATA.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT OUT-FILE ASSIGN TO "employee.dat"
           ORGANIZATION IS LINE SEQUENTIAL.
       DATA DIVISION.
       FILE SECTION.
       FD  OUT-FILE.
       01  OUT-REC.
           05 EMP-ID PIC 9(05).
           05 EMP-NAME PIC X(20).
           05 EMP-DEPT PIC X(10).

       WORKING-STORAGE SECTION.
       01  WS-EMP-REC.
           05 WS-EMP-ID PIC 9(05).
           05 WS-EMP-NAME PIC X(20).
           05 WS-EMP-DEPT PIC X(10).

       LINKAGE SECTION.
       01  LK-EMP-REC.
           05 LK-EMP-ID PIC 9(05).
           05 LK-EMP-NAME PIC X(20).
           05 LK-EMP-DEPT PIC X(10).
       PROCEDURE DIVISION USING LK-EMP-REC.
           MOVE LK-EMP-REC TO WS-EMP-REC.

           OPEN OUTPUT OUT-FILE.
           MOVE WS-EMP-REC TO OUT-REC.
           WRITE OUT-REC.
           CLOSE OUT-FILE.
           GOBACK.

```

*   **Commentary:** This example passes a structured record (`LK-EMP-REC`) representing employee data. The linkage data matches the structure declared in the file's record layout. The code demonstrates a direct record-to-record copy before writing to the sequential output file "employee.dat". This is a common practice when handling data where the structure is known. The `MOVE` statement here copies the complete record structure.

**Example 3: Table of String Data**

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. WRITE-TABLE-DATA.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT OUT-FILE ASSIGN TO "product.csv"
           ORGANIZATION IS LINE SEQUENTIAL.
       DATA DIVISION.
       FILE SECTION.
       FD  OUT-FILE.
       01  OUT-REC PIC X(80).

       WORKING-STORAGE SECTION.
       01  WS-PRODUCT-TABLE.
           05 WS-PRODUCT-ROW OCCURS 5 TIMES.
              10 WS-PRODUCT-NAME PIC X(20).
              10 WS-PRODUCT-PRICE PIC 9(05)V99.
       01  WS-OUTPUT-LINE PIC X(80).

       LINKAGE SECTION.
       01  LK-PRODUCT-TABLE.
           05 LK-PRODUCT-ROW OCCURS 5 TIMES.
             10 LK-PRODUCT-NAME PIC X(20).
             10 LK-PRODUCT-PRICE PIC 9(05)V99.

       PROCEDURE DIVISION USING LK-PRODUCT-TABLE.
           MOVE LK-PRODUCT-TABLE TO WS-PRODUCT-TABLE.
           OPEN OUTPUT OUT-FILE.
           PERFORM VARYING I FROM 1 BY 1 UNTIL I > 5
               STRING
                   WS-PRODUCT-NAME(I) "," WS-PRODUCT-PRICE(I)
                   DELIMITED BY SIZE INTO WS-OUTPUT-LINE
                WRITE OUT-REC FROM WS-OUTPUT-LINE
               END-PERFORM
           CLOSE OUT-FILE.
           GOBACK.
```

*   **Commentary:** Here, a table of data (product names and prices) is passed to the module. The `OCCURS` clause in the `LINKAGE SECTION` defines the table structure. A `PERFORM VARYING` loop is used to iterate through each element of the table and format it as a comma-separated string before writing to the "product.csv" output file.  This demonstrates the handling of repeating data structures.

**Resource Recommendations**

To further solidify these concepts, consider studying resources focused on the following topics:

*   **COBOL Programming Manuals:** Consult vendor-specific documentation for the COBOL compiler you are using.  These usually have comprehensive explanations of language syntax, file handling, and program structuring.
*   **COBOL File I/O:** Deepen your understanding of various file organizations like line sequential, sequential and indexed files and their respective access mechanisms, including writing and updating, is foundational.
*   **COBOL Parameter Passing:** Concentrate on the `LINKAGE SECTION` and parameter passing conventions, including By-Reference and By-Content options. Understand how data is shared between programs and modules.
*   **COBOL Data Structures:** Further study `OCCURS` clause for handling arrays and structures and how they are handled through record definitions and record layouts.
*   **COBOL String Manipulation:** Master the use of statements such as `STRING`, `UNSTRING` and other functions for data formatting.

By applying these techniques and focusing on a systematic approach to data handling, one can effectively pass external variable values into COBOL modules and write them to new output files, enabling seamless integration with various systems.
