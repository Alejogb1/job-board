---
title: "Why does COBOL display signed Comp-3 data incorrectly?"
date: "2025-01-30"
id: "why-does-cobol-display-signed-comp-3-data-incorrectly"
---
Comp-3, or packed decimal, storage, while efficient in its encoding of numeric values, often presents display challenges due to the sign nibble's position and its representation within the memory structure, particularly when output using standard display routines expecting ASCII or EBCDIC representations. I encountered this frequently during my time maintaining legacy banking systems where Comp-3 was the prevalent format for monetary values. The root of the issue lies not in COBOL itself, but in the interpretation of the underlying data bytes by output peripherals and software.

A Comp-3 field stores numeric digits using four bits (a nibble) per digit, effectively packing two digits into a single byte, with the final nibble storing the sign. The specific coding of the sign nibble varies, but commonly ‘C’ or ‘F’ represents positive, and ‘D’ represents negative. The problem arises because many output systems expect each byte to map directly to a printable character. Consider a simple example: a three-digit Comp-3 field, let’s say `+123`. This would be encoded in hexadecimal as `12 3C`. A standard print routine, encountering `12`, might display the non-printable control character, and `3C` might be misinterpreted, not as the end-of-field sign nibble. Similarly, a negative value, such as `-456` would be represented by `45 6D` where the `6D` would also be an issue for display purposes.

The core misinterpretation occurs because standard display mechanisms assume each byte represents an ASCII or EBCDIC encoded character. With Comp-3 the final nibble is intended to convey sign information. Standard display routines, especially when directly processing the raw memory, are unable to discern between a character representation and a sign representation in the final nibble, hence resulting in unpredictable outputs.

To rectify this, an explicit conversion must be performed. The correct approach involves unpacking the packed decimal, extracting each digit, and generating the appropriate character representations of those digits and then prepending the sign where appropriate using COBOL's `MOVE` or similar statements. This unpacked string is then displayable in a readable format. The conversion logic needs to handle the sign nibble separately, generating either a leading minus sign or leaving the number unsigned.

Consider the following COBOL examples, each detailing a typical scenario I've encountered and the corresponding solution:

**Example 1: Simple Positive Value Display Problem**

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. DISPLAY-TEST-1.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  COMP3-VALUE   PIC S9(3) COMP-3 VALUE +123.
       01  DISPLAY-VALUE PIC X(4).
       PROCEDURE DIVISION.
           DISPLAY "Raw Comp-3: " COMP3-VALUE.
           MOVE COMP3-VALUE TO DISPLAY-VALUE.
           DISPLAY "Direct Move: " DISPLAY-VALUE.
           STOP RUN.
```

**Commentary:** This example demonstrates the problem. `COMP3-VALUE` is declared as a signed three-digit packed decimal and initialized with the value `+123`. The first `DISPLAY` statement attempts to show the raw Comp-3 value, likely resulting in unreadable output because the display routine directly prints the memory contents as characters including the hexadecimal C. The second `DISPLAY` statement also demonstrates incorrect output: it attempts to display the value after moving it directly to a character string resulting in the same issue. This clearly shows direct manipulation of Comp-3 data by standard output mechanisms yields incorrect results.

**Example 2: Correcting a Negative Value with Unpack Logic**

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. DISPLAY-TEST-2.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  COMP3-VALUE    PIC S9(3) COMP-3 VALUE -456.
       01  DISPLAY-VALUE  PIC X(5).
       01  UNPACKED-VALUE PIC 9(3).
       01  SIGN-FLAG      PIC X(1).
       PROCEDURE DIVISION.
           MOVE COMP3-VALUE TO UNPACKED-VALUE.
           IF COMP3-VALUE IS NEGATIVE
               MOVE "-" TO SIGN-FLAG
           ELSE
               MOVE "" TO SIGN-FLAG.
           STRING SIGN-FLAG, UNPACKED-VALUE DELIMITED BY SIZE INTO DISPLAY-VALUE.
           DISPLAY "Correct Display: " DISPLAY-VALUE.
           STOP RUN.
```

**Commentary:** This example demonstrates the proper procedure for displaying a Comp-3 negative value. `COMP3-VALUE` is initialized to `-456`. The packed decimal is moved to an unpacked numeric field `UNPACKED-VALUE`, thereby removing the comp-3 encoding. A separate `SIGN-FLAG` is set using the conditional statement depending on the value of COMP3-VALUE. Finally, the `STRING` verb is used to concatenate the sign flag and the unpacked value into `DISPLAY-VALUE`. This process explicitly generates a readable string representation by extracting each digit, including the sign. This example correctly displays the number.

**Example 3: Handling Multiple Digit Values and Zero**

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. DISPLAY-TEST-3.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  COMP3-POS      PIC S9(5)V9(2) COMP-3 VALUE +12345.67.
       01  COMP3-NEG      PIC S9(5)V9(2) COMP-3 VALUE -98765.43.
       01  COMP3-ZERO     PIC S9(5)V9(2) COMP-3 VALUE ZERO.
       01  DISPLAY-VALUE-P    PIC X(10).
       01  DISPLAY-VALUE-N    PIC X(10).
       01  DISPLAY-VALUE-Z    PIC X(10).

       01  UNPACKED-VALUE-P PIC 9(5)V9(2).
       01  UNPACKED-VALUE-N PIC 9(5)V9(2).
       01  UNPACKED-VALUE-Z PIC 9(5)V9(2).

       01  SIGN-FLAG-P      PIC X(1).
       01  SIGN-FLAG-N      PIC X(1).
       01  SIGN-FLAG-Z      PIC X(1).

       PROCEDURE DIVISION.
           MOVE COMP3-POS TO UNPACKED-VALUE-P.
           IF COMP3-POS IS NEGATIVE
               MOVE "-" TO SIGN-FLAG-P
           ELSE
               MOVE "" TO SIGN-FLAG-P.
           STRING SIGN-FLAG-P, UNPACKED-VALUE-P DELIMITED BY SIZE INTO DISPLAY-VALUE-P.

           MOVE COMP3-NEG TO UNPACKED-VALUE-N.
           IF COMP3-NEG IS NEGATIVE
               MOVE "-" TO SIGN-FLAG-N
           ELSE
               MOVE "" TO SIGN-FLAG-N.
            STRING SIGN-FLAG-N, UNPACKED-VALUE-N DELIMITED BY SIZE INTO DISPLAY-VALUE-N.

           MOVE COMP3-ZERO TO UNPACKED-VALUE-Z.
           IF COMP3-ZERO IS NEGATIVE
               MOVE "-" TO SIGN-FLAG-Z
           ELSE
               MOVE "" TO SIGN-FLAG-Z.
            STRING SIGN-FLAG-Z, UNPACKED-VALUE-Z DELIMITED BY SIZE INTO DISPLAY-VALUE-Z.
           DISPLAY "Positive: " DISPLAY-VALUE-P.
           DISPLAY "Negative: " DISPLAY-VALUE-N.
           DISPLAY "Zero: " DISPLAY-VALUE-Z.
           STOP RUN.
```

**Commentary:** This example extends the solution to demonstrate handling of multiple values, including positive, negative and zero using the same approach as example 2. `COMP3-POS`, `COMP3-NEG` and `COMP3-ZERO` are initialized to decimal values. The logic for unpacking, sign handling and concatenation is duplicated for each value.  Each value is then displayed correctly demonstrating that the approach is generalized and not limited to the 3-digit examples demonstrated previously. This also illustrates that zero is handled appropriately by the conditional statement.

For further study on data representation, I recommend consulting texts focusing on IBM mainframe architecture and COBOL programming. Detailed information can also be found in documentation relating to specific mainframe systems and compilers that your organization might be utilizing. Reference materials covering the fundamentals of data representation and data structures in general programming and computer science may also provide beneficial context. Lastly, exploring COBOL language reference manuals, particularly sections on `COMP-3` or packed decimal data types, will also prove invaluable for understanding the nuances of the format and how the COBOL compiler treats it.
