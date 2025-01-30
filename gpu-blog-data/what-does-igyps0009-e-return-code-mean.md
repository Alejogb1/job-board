---
title: "What does IGYPS0009-E return code mean?"
date: "2025-01-30"
id: "what-does-igyps0009-e-return-code-mean"
---
The IGYPS0009-E return code, frequently encountered during the compilation of COBOL programs under the IBM Enterprise COBOL for z/OS compiler, specifically indicates a failure related to the handling of a user-defined function. This error arises during the semantic analysis phase of compilation, specifically when the compiler encounters an issue within the parameters or invocation of a function explicitly coded by the programmer. This isn’t a system-level failure, but rather a result of incorrect coding practices when designing and implementing user-defined COBOL functions. I have debugged numerous complex batch processes over the past decade, and this code, while not infrequent, always signals a critical need to scrutinize the function definition and call sites.

Specifically, IGYPS0009-E denotes that the compiler has detected an inconsistency between the function definition, as specified in the `FUNCTION-ID` paragraph, and how the function is being invoked (called) within the program. This inconsistency manifests primarily in three forms: an incorrect number of arguments, incompatible data types of arguments passed versus parameters defined, or a lack of a required return data type. Essentially, the compiler is flagging a misuse of the contract between the function's interface and its usage.

Let's break down the common scenarios I've directly encountered, alongside representative code snippets to illustrate.

**Scenario 1: Incorrect Number of Arguments**

This is the most common manifestation of IGYPS0009-E. Consider a scenario where I defined a function to calculate the sum of two integers.

```cobol
       IDENTIFICATION DIVISION.
       FUNCTION-ID. SUM-TWO-INTS.
       DATA DIVISION.
       LINKAGE SECTION.
       01  INT-A           PIC S9(9) BINARY.
       01  INT-B           PIC S9(9) BINARY.
       PROCEDURE DIVISION USING INT-A, INT-B RETURNING RET-VAL.
       01  RET-VAL         PIC S9(9) BINARY.
           COMPUTE RET-VAL = INT-A + INT-B
           EXIT FUNCTION.
```

The function `SUM-TWO-INTS` explicitly expects two integer arguments. However, if the calling program attempts to invoke it with a different number of arguments, for instance one or three, this results in an IGYPS0009-E error.

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. EXAMPLE-PROG.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  VAR-X           PIC S9(9) BINARY VALUE 5.
       01  VAR-Y           PIC S9(9) BINARY VALUE 10.
       01  VAR-Z           PIC S9(9) BINARY VALUE 15.
       01  SUM-RES         PIC S9(9) BINARY.
       PROCEDURE DIVISION.
           MOVE FUNCTION SUM-TWO-INTS(VAR-X, VAR-Y, VAR-Z) TO SUM-RES
           DISPLAY "SUM RESULT: " SUM-RES
           STOP RUN.
```

Here, `EXAMPLE-PROG` tries to pass three arguments to `SUM-TWO-INTS`, violating the function's definition. The compiler would stop with an IGYPS0009-E message pointing to the line where `SUM-TWO-INTS` is called. Correct invocation would look like this: `MOVE FUNCTION SUM-TWO-INTS(VAR-X, VAR-Y) TO SUM-RES`.

**Scenario 2: Incompatible Data Types**

Another common reason for this return code involves passing arguments of a different data type than what the function expects. Consider this function designed to process a numeric value alongside a character string.

```cobol
       IDENTIFICATION DIVISION.
       FUNCTION-ID. PROCESS-DATA.
       DATA DIVISION.
       LINKAGE SECTION.
       01  NUM-VAL         PIC S9(9) BINARY.
       01  STR-VAL         PIC X(20).
       PROCEDURE DIVISION USING NUM-VAL, STR-VAL RETURNING RET-VAL.
       01  RET-VAL         PIC X(50).
           STRING "Number is: " NUM-VAL " String is: " STR-VAL DELIMITED BY SIZE INTO RET-VAL
           EXIT FUNCTION.
```

This function expects a numeric `NUM-VAL` and a character string `STR-VAL`. Now, if the calling program sends a character string for the `NUM-VAL` or vice versa, it results in an IGYPS0009-E.

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TYPE-MISMATCH.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  NUMBER-VAL      PIC S9(9) BINARY VALUE 100.
       01  TEXT-VAL        PIC X(20) VALUE "HELLO WORLD".
       01  RESULT-STR      PIC X(50).
       PROCEDURE DIVISION.
           MOVE FUNCTION PROCESS-DATA(TEXT-VAL, NUMBER-VAL) TO RESULT-STR
           DISPLAY "Result: " RESULT-STR
           STOP RUN.
```

In the above code, the `TYPE-MISMATCH` program incorrectly passes `TEXT-VAL` (a string) where `PROCESS-DATA` expects a numeric argument and `NUMBER-VAL` (a number) where `PROCESS-DATA` expects a string. The compiler will flag this incorrect argument order as an IGYPS0009-E error. The correct call should be: `MOVE FUNCTION PROCESS-DATA(NUMBER-VAL, TEXT-VAL) TO RESULT-STR`.

**Scenario 3: Missing or Incorrect Return Data Type**

While less frequently observed, I've encountered cases where the error results from a problem with the function's return data type. Primarily, this stems from either failing to specify `RETURNING` in the `PROCEDURE DIVISION` header of the function, or from invoking a function without handling its returned value. If a function is designed to return a result, but the calling program neglects to assign this result to a variable, this can lead to issues interpreted as part of an IGYPS0009-E context.

```cobol
       IDENTIFICATION DIVISION.
       FUNCTION-ID. NO-RETURN-TYPE.
       DATA DIVISION.
       LINKAGE SECTION.
       01  INPUT-VAL        PIC S9(9) BINARY.
       PROCEDURE DIVISION USING INPUT-VAL.
           COMPUTE INPUT-VAL = INPUT-VAL * 2
           EXIT FUNCTION.
```

The above function does not specify a `RETURNING` clause. If we attempt to use the function in a program expecting a return value, it will generate the IGYPS0009-E.

```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. NO-RETURN-PROGRAM.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  VALUE-TO-DOUBLE PIC S9(9) BINARY VALUE 5.
       01  RESULT-VALUE    PIC S9(9) BINARY.
       PROCEDURE DIVISION.
           MOVE FUNCTION NO-RETURN-TYPE(VALUE-TO-DOUBLE) TO RESULT-VALUE
           DISPLAY "Doubled value is: " RESULT-VALUE
           STOP RUN.
```

The calling program tries to assign the non-existent return value to `RESULT-VALUE`.  To resolve this, the function definition would have to include a `RETURNING` clause and a return value in the `LINKAGE SECTION`. Furthermore, the calling program must correctly handle the return value.

**Recommendations and Troubleshooting**

When confronted with an IGYPS0009-E error, I recommend a systematic approach:

1.  **Carefully Examine the Function Definition:** Scrutinize the `FUNCTION-ID` paragraph, focusing on the `LINKAGE SECTION` and the `PROCEDURE DIVISION USING ... RETURNING ...` clause. Precisely identify the expected number and data types of parameters and the type of the return value.

2. **Review All Call Sites:** Methodically inspect each point in your program where the user-defined function is invoked. Cross-reference the parameters being passed with the function's definition. Ensure the number, types, and order of arguments are consistent. Verify that the return value is correctly assigned.

3. **Use Compiler Listings:** The compiler listing will typically pinpoint the line of code triggering the error, which significantly expedites the debugging. Pay close attention to the compiler-generated messages around the flagged call site.

4.  **Simplify Your Code:** When tackling complex function calls, begin with minimal test cases to isolate the error source. Remove any unnecessary complexity to more easily pinpoint discrepancies.

5. **Consult COBOL References:** The IBM Enterprise COBOL for z/OS Language Reference manual (specifically the sections on User Defined Functions) provides a comprehensive explanation of the required syntax and usage. Also valuable are books dedicated to modern COBOL programming techniques, and online community forums dedicated to COBOL development where developers share their insights and experiences.

In conclusion, the IGYPS0009-E error serves as a precise diagnostic tool for identifying misalignments between function definitions and usage within COBOL programs. Approaching debugging with a meticulous examination of both the function specification and all its invocation points, combined with a thorough understanding of parameter matching, will lead to efficient resolution.
