---
title: "What is the meaning of VSAM KSDS file status 39?"
date: "2025-01-30"
id: "what-is-the-meaning-of-vsam-ksds-file"
---
A VSAM KSDS (Key Sequenced Data Set) file status code of 39 typically indicates a failure during an I/O operation due to insufficient space to allocate a new control interval within the dataset. This often arises when attempting to add a new record to a VSAM file where the current available free space cannot accommodate the new record's size, or related overhead, according to the defined control interval size. Having encountered this numerous times during my tenure working with legacy systems, I’ve come to understand its nuanced implications.

Fundamentally, VSAM (Virtual Storage Access Method) organizes data into control intervals (CIs), which are the basic units of data transfer between auxiliary storage and memory. Within a KSDS, records are stored in a specific order based on their key value. When a new record is written, VSAM looks for available space within an existing CI. If there isn't sufficient space, VSAM will attempt to allocate a new CI. A status code of 39 signifies that this CI allocation has failed. Several factors can contribute to this, including: lack of sufficient primary and secondary space allocation, excessive control area splits that fragment the dataset's space, or an extremely high fill ratio within control intervals during an insert or update.

From a practical standpoint, a VSAM KSDS with status code 39 is operationally impaired. It cannot reliably add new records and may experience performance degradation if it has already reached its primary space limit and has started using secondary extents which cause frequent additional I/Os. Data processing operations that rely on this VSAM dataset will fail. Troubleshooting requires reviewing the VSAM dataset's space allocation, examining the control interval size configuration, and assessing how frequently the dataset is growing. It's not always a simple matter of adding more space; often, performance considerations require adjusting the VSAM parameters for optimum efficiency.

The following three code examples, based on my past experiences, illustrate common scenarios and how to approach them, although specific coding languages and system environment will have their particularities.

**Code Example 1: COBOL Program Handling the Status 39 Error**

```COBOL
       IDENTIFICATION DIVISION.
       PROGRAM-ID. VSAM-INSERT.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT VSAM-FILE ASSIGN TO VSAM-DSN
                  ORGANIZATION IS INDEXED
                  ACCESS MODE IS RANDOM
                  RECORD KEY IS REC-KEY
                  FILE STATUS IS WS-FILE-STATUS.
       DATA DIVISION.
       FILE SECTION.
       FD  VSAM-FILE.
       01  VSAM-REC.
           05 REC-KEY PIC X(10).
           05 REC-DATA PIC X(50).
       WORKING-STORAGE SECTION.
       01  WS-FILE-STATUS    PIC XX.
       01 WS-NEW-REC PIC X(60) VALUE '1234567890DATA VALUE'.
       PROCEDURE DIVISION.
       MAIN-PARA.
           OPEN I-O VSAM-FILE.
           MOVE WS-NEW-REC (1:10) TO REC-KEY.
           MOVE WS-NEW-REC (11:50) TO REC-DATA.
           WRITE VSAM-REC.
           EVALUATE WS-FILE-STATUS
             WHEN "00" DISPLAY "Record Written Successfully"
             WHEN "39" DISPLAY "ERROR: VSAM Space Allocation Failure"
                       DISPLAY "Review VSAM dataset attributes: Space, CI Size"
             WHEN OTHER DISPLAY "Other VSAM Error: ", WS-FILE-STATUS
           END-EVALUATE.
           CLOSE VSAM-FILE.
           STOP RUN.
```

In this COBOL program, `WS-FILE-STATUS` is a critical variable that captures the result of each VSAM I/O operation. After attempting to write a record, an `EVALUATE` statement checks its value. When the value is "39", the code explicitly displays an error message advising the user to review the VSAM dataset's space allocation and CI size, based on the understanding that insufficient space or CI configuration is causing the issue. This practical handling is paramount to understanding VSAM operational issues quickly.

**Code Example 2: JCL for VSAM Dataset Reorganization**

```JCL
//VSAMREOR JOB
//*--------------------------------------------------------------------*
//STEP1    EXEC PGM=IDCAMS
//SYSPRINT DD SYSOUT=*
//SYSIN    DD  *
      REPRO -
        INFILE(OLDVSAM) -
        OUTFILE(NEWVSAM)
      DELETE OLDVSAM CLUSTER
      DEFINE -
        CLUSTER (NAME(OLDVSAM) -
                INDEXED -
                KEYS(10 0) -
                RECORDSIZE(60 60) -
                VOLUMES(VOL001) -
                CYL(50 25) -
                SHAREOPTIONS(3 3) ) -
       DATA   (NAME(OLDVSAM.DATA) ) -
       INDEX  (NAME(OLDVSAM.INDEX) )
/*
//OLDVSAM  DD DSN=YOUR.OLD.VSAM.DSN,DISP=SHR
//NEWVSAM  DD DSN=YOUR.NEW.VSAM.DSN,DISP=(NEW,CATLG,DELETE),
//         SPACE=(CYL,(75,50),RLSE),UNIT=SYSDA
```

This JCL (Job Control Language) snippet outlines the steps required to reorganize a VSAM dataset. The process involves using the IDCAMS utility to perform a repro operation from the old VSAM dataset to a new one (`NEWVSAM`), effectively defragmenting the data. The old VSAM dataset is then deleted, and a new definition is put in place. Crucially, the `DEFINE CLUSTER` command includes an explicit space allocation (`CYL(50 25)`) where more space may be given to avoid the space related status 39. During the process, data is copied to the new dataset, reclaiming any previously unusable fragmented space. While this example does not directly involve the status 39 check, it illustrates how one would address the problem, which is a lack of effective space, through reorganization.

**Code Example 3: REXX Script for Space Verification**

```REXX
/* REXX Script to check VSAM Space */
ADDRESS ISPEXEC
"CONTROL ERRORS RETURN"
"SELECT ISPEDIT"
"EDIT DATASET('VSAM.DATASET.LIST') MEMBER(MEMB1) ' '"
"FIND 'YOUR.VSAM.DSN'  ALL"
IF RC > 0 THEN
   SAY "VSAM Dataset Not found in configuration."
   EXIT 0

DATASET = ''
LINE = 1
DO WHILE  SUBSTR(ZEDIT, 1,2) <> '  '
   PARSE VAR ZEDIT '' DATASET .
   IF DATASET <> ''  THEN ITERATE
   LINE = LINE + 1
   "EDIT LINE" LINE " ' '"
END
ADDRESS TSO

/* Get dataset attributes */
"LISTCAT ENT(DATASET) ALL"
/* process the listcat output and look for space allocation */
/* and return the relevant values for further analysis */
IF RC = 0 THEN
   DO
     "EXECIO * DISKR LISTCAT.OUTPUT (FINIS STEM LIST_."
     DO i=1 TO LIST_.0
         IF POS('ALLOCATION', LIST_.i) > 0 THEN
         PARSE VAR LIST_.i ' ' 5 +10 . ' ' START ' '  .
         IF POS('SPACE-TYPE', LIST_.i) > 0 THEN
         PARSE VAR LIST_.i ' ' 5 +10 . ' ' SPACE_TYPE ' '  .
         IF POS('PRIMARY-SPACE', LIST_.i) > 0 THEN
         PARSE VAR LIST_.i ' ' 5 +10 . ' ' PRIMARY ' ' .
         IF POS('SECONDARY-SPACE', LIST_.i) > 0 THEN
         PARSE VAR LIST_.i ' ' 5 +10 . ' ' SECONDARY ' '  .
    END
      SAY "Space Type: "SPACE_TYPE
      SAY "Primary Space "PRIMARY
      SAY "Secondary Space "SECONDARY
   END
ELSE
    SAY "Error accessing VSAM dataset catalog"
    EXIT 0
EXIT 0
```

This REXX script demonstrates a practical approach to verifying VSAM dataset space information. It uses ISPF (Interactive System Productivity Facility) to identify the VSAM dataset in a configuration file, then calls TSO (Time Sharing Option) to execute a `LISTCAT` command to retrieve dataset attributes, parsing the output to isolate relevant space metrics like `PRIMARY` and `SECONDARY` space. While it doesn't explicitly encounter the status 39, the script helps gather data crucial for diagnosing and preventing its occurrence. It showcases the importance of monitoring VSAM dataset attributes to proactively identify situations that may lead to status 39.

In summary, status code 39 in a VSAM KSDS signifies insufficient space available for a new control interval allocation. Addressing this involves analyzing the current space allocation, assessing the fill factor, and reorganizing the VSAM dataset as necessary. Code examples provided here offer a glimpse of the diagnostic and corrective approaches for identifying and resolving the underlying space-related issues causing the status code 39, whether on the application side, job control language, or system utilities such as REXX.

For further information and deeper understanding of VSAM concepts, I recommend studying the IBM Redbooks on VSAM, specific product manuals for z/OS operating systems, and resources dedicated to VSAM performance management from specialized technical publishers. Focusing on topics like VSAM access method services, dataset definition parameters, and efficient VSAM file organization is essential for properly handling problems like status code 39.
