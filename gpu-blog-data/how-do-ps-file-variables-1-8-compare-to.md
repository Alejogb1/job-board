---
title: "How do PS file variables 1-8 compare to PDS members 1-8 to create a PDS member file for matching PDS names?"
date: "2025-01-30"
id: "how-do-ps-file-variables-1-8-compare-to"
---
The core discrepancy between PS file variables and PDS members lies in their fundamental nature: PS files are sequential datasets, essentially flat files, while PDS members are individually addressable components within a partitioned dataset.  Direct comparison of variables 1-8 from a PS file to identically numbered members in a PDS is therefore not a straightforward process;  it necessitates a structured approach involving data extraction, transformation, and recreation within the target PDS. In my experience working with mainframe data migration projects, this type of transformation is common, often requiring careful consideration of record structures and potential data discrepancies.

My approach consistently involves a three-stage process: data extraction from the PS file, data transformation based on the requirements for the PDS members, and finally, the generation of the PDS members from the transformed data.  I'll illustrate this process with code examples using JCL and REXX.  While the specific syntax and utility usage may vary based on the mainframe environment (z/OS, etc.), the underlying principles remain consistent.

**1. Data Extraction:**

The first step requires extracting the relevant data from the PS file.  This commonly involves using a utility like DFSORT.  DFSORT provides powerful data manipulation capabilities, allowing us to select specific fields and reformat data according to our needs.  We will assume that the PS file contains records structured such that variables 1-8 are contiguous fields within each record.  The exact specifications for variable extraction will depend on the record format of the PS file, likely indicated by a record format description (RFD) or similar documentation.

```jcl
//STEP1  EXEC PGM=SORT
//SYSOUT DD SYSOUT=*
//SORTIN DD DISP=SHR,DSN=PS.FILE.NAME
//SORTOUT DD DSN=EXTRACTED.DATA,DISP=(,CATLG),
//            SPACE=(CYL,(10,10),RLSE),DCB=(LRECL=80,RECFM=FB)
//SYSIN DD *
  OPTION COPY
  INREC IFTHEN=(1,8,CH,EQ,C'ABC',IF=(1,10,CH,OUTREC)) /*Example condition to filter*/
  OUTREC  BUILD=(1,10,1,8,10,10,10,10,10,10,10,10) /*Example of fields to extract*/
/*
 Replace the placeholder 'ABC' with your conditional field value, and adjust the INREC/OUTREC according to the actual variables' lengths and positions.
*/
```

This JCL job uses DFSORT to read the PS file (`PS.FILE.NAME`).  The `INREC` and `OUTREC` statements define how the input records are processed and what's written to the output dataset (`EXTRACTED.DATA`).  The example includes a simple conditional check and a structured `BUILD` statement to copy specific fields (variables 1-8, assuming they occupy the first 80 bytes).  Adapt these parameters according to your PS file's structure. This extracted data is then used as the basis for creating the PDS members.


**2. Data Transformation:**

Rarely does the extracted data directly mirror the required format of the PDS members.  This often necessitates transformation, which can involve data type conversions, field reordering, or other manipulations.  REXX is particularly well-suited for this stage.  It enables scripting to handle complex data transformations and create the necessary control statements for subsequent PDS member creation.


```rexx
/* REXX program to transform extracted data */
address TSO
"ALLOC FI(INFILE) DA('EXTRACTED.DATA') SHR REUSE"
"ALLOC FI(OUTFILE) DA('PDS.MEMBER.DATA') NEW REUSE"
do while lines(infile) > 0
  line = linein(infile)
  /*Process each line from the extracted data*/
  var1 = substr(line,1,10) /*Extract variable 1*/
  var2 = substr(line,11,10) /*Extract variable 2*/
  /* ... extract other variables */
  transformed_line = var1 || ',' || var2 || /*... concatenate variables, add separators*/
  "WRITE OUTFILE " transformed_line
end
"FREE FI(INFILE)"
"FREE FI(OUTFILE)"
```

This REXX program reads the extracted data (`EXTRACTED.DATA`), processes each line, extracts relevant variables (adjusting substrings as needed based on field lengths and positions), performs necessary transformations, and writes the transformed data to a temporary file (`PDS.MEMBER.DATA`).  The concatenation uses separators to match the expected PDS member format (e.g., comma-separated values).  Error handling and data validation should be added for robustness in a production environment.

**3. PDS Member Creation:**

The final step involves using the transformed data to create the individual PDS members.  This requires JCL again, often employing utility programs like IEBCOPY or IDCAMS.  IEBCOPY is simpler for straightforward copy operations, while IDCAMS provides more advanced dataset manipulation capabilities.


```jcl
//STEP3 EXEC PGM=IEBCOPY
//SYSPRINT DD SYSOUT=*
//SYSIN DD *
 COPY INDATASET=PDS.MEMBER.DATA OUTDATASET=PDS.NAME(MEMBER1)
 COPY INDATASET=PDS.MEMBER.DATA OUTDATASET=PDS.NAME(MEMBER2)
/*...repeat for other members*/
/*
   Replace PDS.NAME with the actual PDS name, and MEMBER1, MEMBER2... with the desired member names.
   The number of COPY statements should match the number of PDS members to be created.
   Additional options might be needed for record formatting and data validation, depending on the specific requirements.
*/
```

This JCL job uses IEBCOPY to create multiple PDS members within the target PDS (`PDS.NAME`).  Each `COPY` statement specifies the input dataset (`PDS.MEMBER.DATA`) and the target PDS member name (MEMBER1, MEMBER2, etc.).  Each member will contain one transformed record (or more depending on the transformation's output).  For larger transformations, dynamic JCL generation via a REXX script might be more efficient.


**Resource Recommendations:**

For detailed information on JCL, DFSORT, REXX, and IDCAMS, consult your mainframe system's documentation and available tutorials.  These resources provide comprehensive explanations of the syntax, options, and usage of these tools.  Furthermore, studying examples within your specific mainframe environment will prove invaluable.  Thoroughly testing the solution with sample data and comparing results with expected outputs is crucial before implementing it in a production setting.  Consider using a test PDS to avoid unintended modifications to production data.  Detailed log analysis during each step will be instrumental in troubleshooting and enhancing the process.  Finally, understand the potential for data truncation or errors based on the varying lengths of the data extracted from the PS file and the space allocated for each PDS member.  Rigorous data validation throughout the process is paramount.
