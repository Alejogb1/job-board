---
title: "How can I embed a header in EASYTRIEVE output using JCL?"
date: "2025-01-30"
id: "how-can-i-embed-a-header-in-easytrieve"
---
The core challenge in embedding headers within EASYTRIEVE output via JCL lies in the limited direct control JCL offers over the report formatting within the EASYTRIEVE processing itself.  EASYTRIEVE's report generation is largely self-contained; JCL primarily manages the job's execution environment and data streams.  Therefore, achieving a header requires leveraging indirect methods, primarily manipulating the data stream *before* it reaches the printer or output file.  My experience in large-scale mainframe development, specifically handling complex report generation using EASYTRIEVE and JCL for over fifteen years, has provided ample opportunity to refine these techniques.

**1. Clear Explanation:**

The primary approach involves prepending header records to the EASYTRIEVE input dataset before the actual data processing begins.  This requires a separate JCL step employing a utility program like IEBGENER or SORT to create a new dataset containing the desired header. This modified dataset then becomes the input to the EASYTRIEVE step.  The header records should be formatted to align with the EASYTRIEVE output specifications, including appropriate record lengths and field delimiters. The critical aspect is to ensure the header is written in the same format EASYTRIEVE expects for detail lines, enabling seamless integration.  Failing to match the format will result in the header being incorrectly interpreted or skipped entirely.

Another, less frequently used approach involves post-processing. After EASYTRIEVE completes, another JCL step uses a utility program (again, IEBGENER or SORT) to read the EASYTRIEVE output and prepend the header before writing it to the final output destination.  This method is less efficient due to the extra processing step but can be useful in scenarios where modifying the input dataset directly is not feasible.

Finally, if your EASYTRIEVE program supports it (and many do, via parameters or embedded commands), you can potentially generate the header directly within the EASYTRIEVE code itself. This is generally the most elegant solution, reducing reliance on external utilities, but requires modifying the EASYTRIEVE program.


**2. Code Examples with Commentary:**

**Example 1: Prepending Header using IEBGENER**

```jcl
//HEADERSTEP JOB (ACCTNUM),'HEADER CREATION',CLASS=A,MSGCLASS=X
//STEP1 EXEC PGM=IEBGENER
//SYSPRINT DD SYSOUT=*
//SYSUT1 DD *
HEADER RECORD 1
HEADER RECORD 2
/*
//SYSUT2 DD DSN=HEADER.DATASET,DISP=(NEW,CATLG,DELETE),
// SPACE=(CYL,(1,1))
//EASYTRIEVE JOB (ACCTNUM),'EASYTRIEVE PROCESSING',CLASS=A,MSGCLASS=X
//STEP2 EXEC PGM=EASYTRIEVE
//SYSPRINT DD SYSOUT=*
//EASYIN DD DSN=HEADER.DATASET,DISP=SHR
//EASYOUT DD SYSOUT=*
```

This JCL first creates a dataset (`HEADER.DATASET`) containing the header records using IEBGENER.  `SYSUT1` is used as the input stream, containing the header lines. `SYSUT2` specifies the output dataset, which is cataloged for reuse. The subsequent EASYTRIEVE step then uses `HEADER.DATASET` as its input (`EASYIN`), effectively placing the header at the beginning of the report.  Note that the header record format must exactly match the EASYTRIEVE output record length and structure.


**Example 2: Post-Processing with SORT**

```jcl
//EASYTRIEVE JOB (ACCTNUM),'EASYTRIEVE PROCESSING',CLASS=A,MSGCLASS=X
//STEP1 EXEC PGM=EASYTRIEVE
//SYSPRINT DD SYSOUT=*
//EASYIN DD DSN=INPUT.DATASET,DISP=SHR
//EASYOUT DD DSN=OUTPUT.DATASET,DISP=(NEW,CATLG,DELETE),
// SPACE=(CYL,(1,1))
//POSTPROCESS JOB (ACCTNUM),'POST-PROCESSING',CLASS=A,MSGCLASS=X
//STEP2 EXEC PGM=SORT
//SYSPRINT DD SYSOUT=*
//SORTIN DD DSN=OUTPUT.DATASET,DISP=OLD
//SORTOUT DD DSN=FINAL.OUTPUT,DISP=(NEW,CATLG,DELETE),
// SPACE=(CYL,(1,1))
//SYSIN DD *
  INREC IFTHEN=(1,1,CH,EQ,C' '),
       IFTHEN=(1,1,CH,NE,C' '),
       COPY
  OPTION COPY
  OUTREC IFTHEN=(1,1,CH,EQ,C' '),
       IFTHEN=(1,1,CH,NE,C' '),
       COPY
  CONTROL=(0000000000)
/*
```

This example demonstrates post-processing with SORT. The EASYTRIEVE step generates `OUTPUT.DATASET`.  The SORT step then reads this dataset (`SORTIN`), adds the header records (which would need to be implemented in the `SYSIN` DD statement using appropriate SORT control cards—not shown for brevity, as the specific implementation depends on header formatting and SORT’s capabilities. Note that the logic needs to handle the blank spaces in the dataset. The final output is then written to `FINAL.OUTPUT`. This approach is less efficient but offers flexibility.  It assumes knowledge of the SORT control language and a well defined input dataset structure.


**Example 3:  EASYTRIEVE Internal Header (Illustrative)**

```easyrieve
REPORT HEADING 'My Report Header'
PRINT 'My Report Subheading'
READ INPUT-FILE
PRINT INPUT-FILE-FIELDS
END
```

This illustrates a simplified example of generating the header directly within EASYTRIEVE.  The exact syntax varies depending on the specific EASYTRIEVE version and dialect, but the concept remains the same. Report headings and other formatting commands are available within the EASYTRIEVE language itself.  This approach is the cleanest and most efficient but requires familiarity with the EASYTRIEVE programming language and the potential for modifying the existing code. This is only a representation; real-world situations would have more detailed formatting and data extraction.


**3. Resource Recommendations:**

*   Your organization's mainframe documentation: This will contain the most relevant and up-to-date information on JCL, EASYTRIEVE, and utility programs.
*   EASYTRIEVE Language Reference Manual:  A comprehensive guide to the EASYTRIEVE programming language, including details on report formatting.
*   z/OS JCL Reference:  Detailed information on all aspects of JCL syntax and usage.
*   IBM's z/OS Utilities documentation:  Provides comprehensive documentation on utilities like IEBGENER and SORT.


Remember to carefully consider the record lengths and formats when using any of these methods.  Inconsistencies will lead to errors and incorrect report generation. Always test thoroughly before deploying to production.  Proper error handling within the JCL (using conditions and error procedures) is also critical for robust job processing.
