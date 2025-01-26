---
title: "What is COBOL file status 98?"
date: "2025-01-26"
id: "what-is-cobol-file-status-98"
---

COBOL file status 98 indicates a specific error condition: an invalid record key was encountered during an indexed file operation. This situation arises when a program attempts to locate or modify a record in an indexed file using a key value that is not defined within the file's index structure or is otherwise invalid for the file's current state. I’ve encountered this frequently in legacy mainframe environments, particularly when working with poorly maintained data dictionaries or inadequate validation procedures. It's often the result of a mismatch between how a program interprets a key and how the file's index is actually structured.

The occurrence of file status 98 is usually triggered during I/O operations, most commonly READ, WRITE, REWRITE, and DELETE statements, when these statements involve indexed access. Unlike sequential file operations, indexed file operations rely heavily on an index to quickly locate records based on a designated key. The index itself is a structure of key values and associated record addresses, and any attempt to utilize a non-existent or corrupt key entry results in the program setting file status 98.

The root causes of this error can vary but generally fall into several categories. First, the key value being used may not exist in the index, either because the desired record was never written or because it was deleted earlier. Second, the structure of the key might be defined differently in the program versus how it's actually stored in the file, often resulting in subtle byte-level mismatches. Consider the case of a key defined as alphanumeric in one program, and numeric in another, while the file actually uses a binary representation. Third, the index itself could be corrupted, either due to underlying hardware issues, software bugs in the file manager, or incorrect recovery procedures. Finally, concurrent access to the indexed file without adequate locking mechanisms can, in rare cases, result in index corruption leading to file status 98.

Here are some practical examples based on my experience dealing with this:

**Example 1: Incorrect Key Value**

Consider a scenario involving a customer file keyed on customer ID. The file is indexed on the customer ID field, which is assumed to be a numeric field, `CUST-ID PIC 9(6)`. The following code attempts to read a record using a hardcoded invalid customer ID:

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. EXAMPLE1.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE
               ASSIGN TO 'CUSTOMER.DAT'
               ORGANIZATION IS INDEXED
               ACCESS IS RANDOM
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-FILE-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05 CUST-ID PIC 9(6).
           05 CUST-NAME PIC X(30).
           05 CUST-ADDR PIC X(50).
       WORKING-STORAGE SECTION.
       01  WS-FILE-STATUS PIC XX.
       PROCEDURE DIVISION.
       MAIN-PARA.
           OPEN I-O CUSTOMER-FILE.
           MOVE 999999 TO CUST-ID.
           READ CUSTOMER-FILE
               INVALID KEY
                   DISPLAY 'Record not found for key ' CUST-ID
           END-READ.
           IF WS-FILE-STATUS = '98'
              DISPLAY 'File Status is 98 for key ' CUST-ID
           END-IF.
           CLOSE CUSTOMER-FILE.
           STOP RUN.
```

*Commentary:* In this example, the `READ` operation attempts to locate a record using a key value of `999999`. If this customer ID does not exist in the indexed file, the `INVALID KEY` clause is not executed (this clause handles situations like '02' or '23' statuses). The FILE STATUS field, `WS-FILE-STATUS`, is updated to '98' after the READ. The conditional statement then detects file status 98, and a message is displayed, indicating that a key not in the index was used. It's crucial to monitor this after each READ, WRITE, REWRITE or DELETE when the file is indexed.

**Example 2: Incorrect Key Structure**

Here, the problem is not the key *value* itself, but how the key is *defined*. Assume the `CUSTOMER-FILE` index is defined using a packed decimal format for the `CUST-ID`, internally. The application code, however, incorrectly defines it as a regular numeric field:

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. EXAMPLE2.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE
               ASSIGN TO 'CUSTOMER.DAT'
               ORGANIZATION IS INDEXED
               ACCESS IS RANDOM
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-FILE-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05 CUST-ID PIC 9(6).
           05 CUST-NAME PIC X(30).
           05 CUST-ADDR PIC X(50).
       WORKING-STORAGE SECTION.
       01  WS-FILE-STATUS PIC XX.
       01  WS-KEY-INPUT PIC 9(6) VALUE 123456.
       PROCEDURE DIVISION.
       MAIN-PARA.
           OPEN I-O CUSTOMER-FILE.
           MOVE WS-KEY-INPUT TO CUST-ID.
           READ CUSTOMER-FILE
               INVALID KEY
                   DISPLAY 'Record not found for key ' CUST-ID
           END-READ.
          IF WS-FILE-STATUS = '98'
             DISPLAY 'File status 98 because key structure is different.'
          END-IF.
           CLOSE CUSTOMER-FILE.
           STOP RUN.
```
*Commentary:* The customer ID within the `CUSTOMER-RECORD` is declared as `PIC 9(6)`, implying an unpacked representation (like ASCII). However, the index, when built, is defined on the physical data, which is a packed decimal version of the same. Even when passing the valid customer number, the underlying storage will not correspond to what the index expected. The `READ` will fail, and file status 98 will be raised. This illustrates a critical point: the key definition in the file structure must precisely match how the index was created. Debugging this often requires examining the file definitions and storage mappings, something that isn't often accessible without specific tooling.

**Example 3: Index Corruption**

While less frequent, the index itself might become corrupted. This scenario is difficult to reproduce predictably in a standalone example. To simulate it, assume that an error occurred, and the index file is partially or completely overwritten with invalid values. Then using a *valid* record key will still result in file status 98:
```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. EXAMPLE3.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE
               ASSIGN TO 'CUSTOMER.DAT'
               ORGANIZATION IS INDEXED
               ACCESS IS RANDOM
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-FILE-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
       01  CUSTOMER-RECORD.
           05 CUST-ID PIC 9(6).
           05 CUST-NAME PIC X(30).
           05 CUST-ADDR PIC X(50).
       WORKING-STORAGE SECTION.
       01  WS-FILE-STATUS PIC XX.
       01 WS-VALID-KEY PIC 9(6) VALUE 123456.
       PROCEDURE DIVISION.
       MAIN-PARA.
           OPEN I-O CUSTOMER-FILE.
           MOVE WS-VALID-KEY TO CUST-ID.
           READ CUSTOMER-FILE
               INVALID KEY
                   DISPLAY 'Record not found for key ' CUST-ID
           END-READ.
          IF WS-FILE-STATUS = '98'
             DISPLAY 'File Status is 98. Index could be corrupted.'
          END-IF.
           CLOSE CUSTOMER-FILE.
           STOP RUN.

```

*Commentary:* In this case, the key, `WS-VALID-KEY` (which is assumed to exist), is moved to the `CUST-ID`, and a READ operation is attempted. With a valid key present and the key definitions being consistent, a READ should not result in file status 98, normally. However, due to the corrupted index, the index itself fails to locate the record. The READ operation sets file status to 98, and the corresponding message is displayed. Correcting such an issue will typically require restoring from backups or utilizing specific file repair utilities, which is beyond the scope of this COBOL program. This illustrates that even with the correct program logic and valid key, underlying data corruption can be a root cause.

When debugging file status 98, it’s essential to start by examining the program’s logic and ensuring that the key values being used match the data that exists in the file. I will usually add output and validation points right before I/O calls, using display statements to show the current content of the keys. If that's good, a review of data dictionary documentation, if available, is critical, to ensure the program's key definitions align with how the data is physically stored. Examining any associated file recovery logs and error messages from the file system itself can offer hints about the source of the problem. Furthermore, careful monitoring of I/O activity and file usage, along with employing robust error handling routines, will allow programs to avoid unexpected termination and also provides valuable information for problem resolution.

Resource recommendations for further understanding include: COBOL language reference manuals, specifically those sections related to file handling and indexed I/O operations. Also, the IBM VSAM documentation which covers the structure of indexed files and error messages in considerable detail. Finally, understanding the specific file system used for the indexed file is important, such as DFSMS. Consulting operating system manuals can help when investigating more complex data corruption issues.
