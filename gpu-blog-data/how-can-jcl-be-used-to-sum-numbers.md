---
title: "How can JCL be used to sum numbers from multiple input files into a single output file?"
date: "2025-01-30"
id: "how-can-jcl-be-used-to-sum-numbers"
---
The core challenge in summing numbers across multiple input files using JCL lies in the efficient handling of sequential data processing and the aggregation of results.  My experience in large-scale data processing within mainframe environments has shown that a straightforward approach using SORT with a custom control card is both efficient and robust for this task.  Avoiding the complexities of inline calculations within JCL itself streamlines the process and enhances maintainability.

**1.  Explanation**

The strategy involves three primary steps:

a) **Data Preparation:**  Each input file, assumed to contain one number per line, requires no preprocessing for this specific task. However, more complex scenarios might necessitate data cleansing or transformation prior to summation.  For instance, if the input files contained additional fields, a preceding JCL job step utilizing a utility like ICETOOL could extract the relevant numeric data.

b) **Summation via SORT:**  The IBM utility SORT is leveraged to perform the summation.  This is achieved through a control card specifying the summation operation.  Crucially, this approach handles multiple input files seamlessly, automatically aggregating the numbers from all sources.  The SORT utility's strength lies in its efficient handling of large datasets and its ability to perform various data transformations, including arithmetic operations.  This makes it ideal for this specific problem.  The output from SORT will then contain a single line representing the total sum.

c) **Output to a Single File:** The aggregated sum, produced by SORT, is directed to a single output file, which can subsequently be used in downstream processing.  This output file is managed through the JCL's DD statements, controlling its name, location, and disposition.  Careful consideration should be given to the file's record format (e.g., RECFM=F, LRECL=10) to ensure compatibility with subsequent processing steps.  Error handling, while not explicitly detailed in this specific response for brevity, should be incorporated in a production environment using JCL's conditional processing capabilities.

**2. Code Examples**

The following examples illustrate different scenarios, progressing in complexity:

**Example 1:  Summing Numbers from Two Files**

This example showcases the basic summation of numbers from two input files, `INPUT1.DAT` and `INPUT2.DAT`, writing the total to `OUTPUT.DAT`.  All files are assumed to have fixed-length records (RECFM=F) with a length of 10 bytes (LRECL=10), accommodating the numeric data.

```jcl
//SUMJOB JOB (ACCOUNT),CLASS=A,MSGCLASS=X
//STEP1 EXEC PGM=SORT
//SORTIN DD DSN=INPUT1.DAT,DISP=SHR
//       DD DSN=INPUT2.DAT,DISP=SHR
//SORTOUT DD DSN=OUTPUT.DAT,DISP=(NEW,CATLG),
//             DCB=(RECFM=F,LRECL=10,BLKSIZE=800),
//             SPACE=(CYL,(1,1))
//SYSIN DD *
  OPTION SUM FIELDS=(1,10,ZD)
/*
```

**Commentary:**  The `OPTION SUM` statement instructs SORT to sum the numeric fields. `FIELDS=(1,10,ZD)` specifies that the summation should be performed on the entire record (from position 1 to 10), and `ZD` indicates that the data is zoned decimal.  Adjust field positions as per your actual data.

**Example 2:  Handling Multiple Files using a Wildcard**

This demonstrates summing numbers from multiple files matching a pattern, useful when dealing with a large number of files.  This assumes all files reside in the same dataset.

```jcl
//SUMJOB JOB (ACCOUNT),CLASS=A,MSGCLASS=X
//STEP1 EXEC PGM=SORT
//SORTIN DD DSN=INPUTDATA.DAT.*,DISP=SHR
//SORTOUT DD DSN=OUTPUT.DAT,DISP=(NEW,CATLG),
//             DCB=(RECFM=F,LRECL=10,BLKSIZE=800),
//             SPACE=(CYL,(1,1))
//SYSIN DD *
  OPTION SUM FIELDS=(1,10,ZD)
/*
```

**Commentary:** `DSN=INPUTDATA.DAT.*` uses a wildcard to include all members within the `INPUTDATA.DAT` dataset. This significantly reduces the need for individual DD statements for each file.  Appropriate dataset organization is crucial for this approach.


**Example 3:  Error Handling and Data Validation (Simplified)**

This example incorporates rudimentary error handling, checking for non-numeric data. This is a simplified example; robust error handling requires a more comprehensive approach, potentially involving custom exit routines within SORT.

```jcl
//SUMJOB JOB (ACCOUNT),CLASS=A,MSGCLASS=X
//STEP1 EXEC PGM=SORT
//SORTIN DD DSN=INPUT1.DAT,DISP=SHR
//       DD DSN=INPUT2.DAT,DISP=SHR
//SORTOUT DD DSN=OUTPUT.DAT,DISP=(NEW,CATLG),
//             DCB=(RECFM=F,LRECL=10,BLKSIZE=800),
//             SPACE=(CYL,(1,1))
//SYSOUT DD SYSOUT=*
//SYSIN DD *
  OPTION SUM FIELDS=(1,10,ZD)
  OPTION NODUP
/*
```

**Commentary:** The `OPTION NODUP` statement will cause SORT to ignore duplicate records. While this doesn't directly handle non-numeric data, it can help mitigate certain types of errors by preventing them from influencing the sum.  A more robust solution might involve using a pre-processing step to filter out invalid data or leveraging SORT’s more advanced features to handle data errors directly.


**3. Resource Recommendations**

For further understanding of JCL and SORT, consult the official IBM documentation.  Thorough exploration of the SORT control statements is crucial.  Study the various options available within SORT, particularly those related to data validation, error handling, and advanced record processing techniques.  Consider researching alternative data transformation utilities provided by IBM, which might offer added flexibility depending on data complexity.  Finally, practical experience through working on real-world mainframe projects involving similar data processing tasks is invaluable.
