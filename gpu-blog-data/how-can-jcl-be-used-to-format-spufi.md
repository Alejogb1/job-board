---
title: "How can JCL be used to format SPUFI results?"
date: "2025-01-30"
id: "how-can-jcl-be-used-to-format-spufi"
---
The ability to format SQL results within batch processing on IBM mainframes using Job Control Language (JCL) and the SQL Processor Using File Input (SPUFI) utility is crucial for generating clear, usable reports and data extracts. I've frequently leveraged this combination in my experience working with legacy systems to produce tailored data outputs for downstream applications. Standard SPUFI output tends to be verbose, including column headers, SQL statement echoes, and extraneous whitespace. Therefore, controlling this output becomes paramount when integrating it with other processes.

The primary mechanism for formatting SPUFI output through JCL involves redirecting the standard output (SYSPRINT) and specifying how the results should be written to a dataset. This process generally utilizes two key elements within the JCL: the SPUFI execution step and the subsequent processing steps. The SPUFI step executes the SQL query, and JCL controls the redirection and structuring of the output. This involves a combination of dataset definitions (DD statements), specific SPUFI control parameters, and potentially, post-processing through other utilities or programs. The strategy hinges on using SPUFI's output controls to minimize unwanted information, and then using JCL datasets to write and organize the remaining data.

Let's consider three practical scenarios.

**Scenario 1: Generating Comma-Separated Values (CSV) Output**

My most common requirement is producing comma-separated files for import into other systems. Here, minimizing the extraneous information from SPUFI output is critical. We need only the data rows, each column separated by commas. To achieve this, we can use the `DELIMITER ,` and `DSNPREF` options within the SPUFI control statements. The `DSNPREF` directs output to a specified dataset name. We also need to disable header information. The JCL example below shows how this can be accomplished:

```jcl
//SPUFIJOB JOB (ACCT),USER,CLASS=A,MSGCLASS=X
//STEP1    EXEC PGM=IKJEFT01
//SYSPRINT DD SYSOUT=*
//SYSTSIN  DD *
  DSN SYSTEM(DB2A)
  RUN PROGRAM(DSNTIAUL) PLAN(DSNTIAUL)
  //SYSIN    DD *
    SELECT EMPNO, FIRSTNME, LASTNAME FROM EMPLOYEE
  //SYSUDUMP DD SYSOUT=*
  //SYSPRINT DD SYSOUT=*
  //SYSOUT   DD SYSOUT=*
  //OUTPUT  DD DSN=USERID.EMP.CSV,DISP=(NEW,CATLG,DELETE),
  //             UNIT=SYSDA,SPACE=(TRK,(1,1),RLSE),
  //             DCB=(RECFM=FB,LRECL=100,BLKSIZE=0)
  //SPUFIIN  DD *
    SET CURRENT SQLID = 'USERID';
    SET CURRENT SCHEMA = 'USERID';
    SELECT 'DELIMITER ,' FROM SYSIBM.SYSDUMMY1;
    SELECT 'DSNPREF OUTPUT' FROM SYSIBM.SYSDUMMY1;
    SELECT EMPNO, FIRSTNME, LASTNAME FROM EMPLOYEE;
    SELECT 'FORMAT OFF' FROM SYSIBM.SYSDUMMY1;
    SELECT 'EXIT' FROM SYSIBM.SYSDUMMY1;
  /*
//
```

*   `//OUTPUT DD`: Defines a new dataset named `USERID.EMP.CSV` where the formatted output will be written. The `DCB` parameters specify the record format as fixed-blocked, a logical record length of 100 bytes, and a system-determined block size.
*   `//SPUFIIN DD`: Defines the input stream for SPUFI. This contains SPUFI control statements along with the actual SQL query.
*   `SET CURRENT SQLID = 'USERID';` and `SET CURRENT SCHEMA = 'USERID';`:  Sets the default schema and SQLID, replacing USERID with the user's ID.
*   `SELECT 'DELIMITER ,' FROM SYSIBM.SYSDUMMY1;` : Sets the delimiter to a comma.
*   `SELECT 'DSNPREF OUTPUT' FROM SYSIBM.SYSDUMMY1;`: Redirects output of the following query to the `OUTPUT` dataset.
*    `SELECT 'FORMAT OFF' FROM SYSIBM.SYSDUMMY1;`:  Suppresses the extraneous SPUFI information, like column headers, for the next query.
*   `SELECT 'EXIT' FROM SYSIBM.SYSDUMMY1;`: Exits SPUFI.

**Scenario 2: Generating a Fixed-Width Report with Header**

In another scenario, I had to produce a report where each column was a fixed width for legacy application compatibility. This required controlling the spacing of each column, along with a header row. This involves leveraging SPUFI to extract the data and additional steps for formatting. The key to formatting is using a combination of `SUBSTR` functions to truncate or pad with spaces, ensuring the output has the correct format, along with generating a header row directly within the JCL, written before the data.

```jcl
//SPUFIJOB JOB (ACCT),USER,CLASS=A,MSGCLASS=X
//STEP1    EXEC PGM=IKJEFT01
//SYSPRINT DD SYSOUT=*
//SYSTSIN  DD *
  DSN SYSTEM(DB2A)
  RUN PROGRAM(DSNTIAUL) PLAN(DSNTIAUL)
  //SYSIN    DD *
    SELECT EMPNO, FIRSTNME, LASTNAME FROM EMPLOYEE
  //SYSUDUMP DD SYSOUT=*
  //SYSPRINT DD SYSOUT=*
  //SYSOUT   DD SYSOUT=*
  //REPORT  DD DSN=USERID.EMP.REPORT,DISP=(NEW,CATLG,DELETE),
  //          UNIT=SYSDA,SPACE=(TRK,(1,1),RLSE),
  //          DCB=(RECFM=FB,LRECL=40,BLKSIZE=0)
  //HDR     DD *,DLM='!!'
  'EMPNO   ','FIRSTNAME','LASTNAME   '!!
  /*
  //STEP2   EXEC PGM=IEBGENER
  //SYSPRINT DD SYSOUT=*
  //SYSUT1   DD DUMMY
  //SYSUT2   DD DSN=USERID.EMP.REPORT,DISP=SHR
  //SYSIN    DD DUMMY
  //SPUFIIN  DD *
  SET CURRENT SQLID = 'USERID';
  SET CURRENT SCHEMA = 'USERID';
  SELECT 'DSNPREF REPORT' FROM SYSIBM.SYSDUMMY1;
  SELECT  SUBSTR(CHAR(EMPNO),1,8),
          SUBSTR(FIRSTNME,1,10),
          SUBSTR(LASTNAME,1,10)
        FROM EMPLOYEE;
  SELECT 'FORMAT OFF' FROM SYSIBM.SYSDUMMY1;
  SELECT 'EXIT' FROM SYSIBM.SYSDUMMY1;
  /*
  //STEP3  EXEC PGM=IEBCOPY
  //SYSPRINT DD SYSOUT=*
  //SYSUT3   DD DUMMY
  //SYSUT4   DD DUMMY
  //SYSIN    DD DUMMY
  //IN1      DD DSN=*.HDR,DISP=SHR
  //OUT1     DD DSN=USERID.EMP.REPORT,DISP=SHR
//
```

*   `//REPORT DD`: Defines the output report dataset similar to scenario 1.
*   `//HDR DD`: Defines an in-stream dataset for the header row. The `DLM='!!'` sets a delimiter to define the end of the header data.
*   `//STEP2 EXEC PGM=IEBGENER`: Uses IEBGENER utility to copy header record to `REPORT` dataset.
*   `//STEP3 EXEC PGM=IEBCOPY`: Uses IEBCOPY utility to merge the header and output data.
*   The SQL query uses `SUBSTR(CHAR(EMPNO),1,8)`, `SUBSTR(FIRSTNME,1,10)`, and `SUBSTR(LASTNAME,1,10)` to extract data and pad each field with spaces up to the specified length. This creates a fixed-width record.
*   The `IEBCOPY` step merges the initial header row from `//HDR` with the dataset `USERID.EMP.REPORT` containing the formatted data.

**Scenario 3: Generating a Data Load File**

Finally, I frequently needed to generate data extract suitable for loading into other systems. This often means producing a file with a specific structure, without delimiters, and possibly using a specific record format. Similar to the fixed-width example, this also involves using `SUBSTR` to format the output, but without headers or delimiters.

```jcl
//SPUFIJOB JOB (ACCT),USER,CLASS=A,MSGCLASS=X
//STEP1    EXEC PGM=IKJEFT01
//SYSPRINT DD SYSOUT=*
//SYSTSIN  DD *
  DSN SYSTEM(DB2A)
  RUN PROGRAM(DSNTIAUL) PLAN(DSNTIAUL)
  //SYSIN    DD *
    SELECT EMPNO, FIRSTNME, LASTNAME FROM EMPLOYEE
  //SYSUDUMP DD SYSOUT=*
  //SYSPRINT DD SYSOUT=*
  //SYSOUT   DD SYSOUT=*
  //LOADFILE DD DSN=USERID.EMP.LOAD,DISP=(NEW,CATLG,DELETE),
  //          UNIT=SYSDA,SPACE=(TRK,(1,1),RLSE),
  //          DCB=(RECFM=FB,LRECL=30,BLKSIZE=0)
  //SPUFIIN  DD *
    SET CURRENT SQLID = 'USERID';
    SET CURRENT SCHEMA = 'USERID';
    SELECT 'DSNPREF LOADFILE' FROM SYSIBM.SYSDUMMY1;
    SELECT  SUBSTR(CHAR(EMPNO),1,8) ||
            SUBSTR(FIRSTNME,1,10) ||
            SUBSTR(LASTNAME,1,12)
        FROM EMPLOYEE;
    SELECT 'FORMAT OFF' FROM SYSIBM.SYSDUMMY1;
    SELECT 'EXIT' FROM SYSIBM.SYSDUMMY1;
  /*
//
```

*   `//LOADFILE DD`: Defines a new output dataset named `USERID.EMP.LOAD` for the load file.
*    The SQL query utilizes the `||` concatenation operator to combine the output of `SUBSTR` functions. This creates a fixed-width record by padding and concatenating strings of specific lengths.
*    No additional steps are needed because the formatted data is written directly to the load file by the SPUFI output.

In all these cases, the choice of dataset characteristics, such as `RECFM`, `LRECL`, and `BLKSIZE`, is critical for ensuring the data is written in the correct format and that downstream processes can access it correctly. Furthermore, the use of `DSNPREF` allows for precise control of which dataset SPUFI output should be directed to.

For further study, I would recommend consulting the IBM documentation for DB2 utilities, specifically the manuals covering SPUFI and JCL. In addition, the materials related to data set attributes (DCB) are paramount. These resources provide detailed information on the parameters and functionalities available for controlling the format and structure of output files. Also, understanding basic JCL syntax and control statements is essential.  The ability to manipulate data through SQL in conjunction with JCL data management provides powerful, flexible capabilities for extracting information from mainframe databases.
