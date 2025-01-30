---
title: "Does changing output file attributes cause erratic looping in a z/OS assembler program?"
date: "2025-01-30"
id: "does-changing-output-file-attributes-cause-erratic-looping"
---
The underlying mechanics of z/OS file handling, specifically with respect to output attributes, can indeed induce unexpected looping behavior within assembler programs. This stems from a potential disconnect between how the program interprets file metadata and how z/OS manages the file's physical characteristics, especially when dynamic allocations are involved. Specifically, issues arise when the program's logic depends on assumptions about file attributes that are altered by system operations outside of the program's direct control, potentially influencing the logical end-of-file (EOF) condition.

A common scenario involves manipulating output data sets via JCL or system utilities while an assembler program simultaneously processes or reads this same file for subsequent operations, such as a read-loop expecting a defined file length. Modifying attributes like record format (RECFM), logical record length (LRECL), or blocksize (BLKSIZE) after the file's initial allocation but before the program has completed its processing can trigger these loops. The assembler program, relying on its initial file control block (FCB) or associated control blocks generated during allocation, might hold a cached version of these attributes. If these attributes change externally, discrepancies arise in interpreting end-of-file conditions that are critical to terminating a loop.

For instance, an assembler program might perform a sequential read using a GET macro within a loop. This loop often relies on a specific return code or condition code following a read operation that indicates an EOF. The program might check for a return code of '2' from the GET macro or examine the standard register-15 for specific return codes that commonly indicate EOF when reading from a file. If the LRECL is increased externally between a series of writes and a subsequent read by the program, the program might encounter short reads or perceive a different end-of-file marker. Alternatively, a change in BLKSIZE could affect the block's boundary, causing the program to miscalculate the number of records processed before reaching end-of-file. These discrepancies lead to the program continuing to read beyond the perceived logical end, thus resulting in an infinite loop or erratic behavior.

Let’s consider a basic program that writes records to a file and then reads them back sequentially. This situation illustrates how attribute changes can be problematic:

**Example 1: Initial Write and Read**

```assembler
         TITLE 'FILEIO EXAMPLE 1'
         PRINT NOGEN
         SYSSTATE AMODE=31
*
*   DEFINE WORK AREAS
         DSECT ,
SAVEAREA DS   18F         Savearea
*
WORKAREA DS  CL100       Work buffer
FILEDCB  DCB   DDNAME=OUTPUT,DSORG=PS,MACRF=PM,       X
               EODAD=EOF1,LRECL=80,BLKSIZE=2792,RECFM=FB
*
         CSECT ,
         USING *,R13
         ST   R13,SAVEAREA+4
         LR    R12,R13
         ST   R12,8(R13)
         LR   R13,SP
*
         OPEN (FILEDCB,OUTPUT)
*
* WRITE SOME SAMPLE RECORDS
         LA   R1,WORKAREA
         MVI  0(R1),C'A'      
         LA  R2,10
LOOP1    STC R2,1(R1)
         PUT  FILEDCB,WORKAREA
         BCT R2,LOOP1
*
*CLOSE OUTPUT
         CLOSE (FILEDCB)
*
         OPEN (FILEDCB,INPUT)
*
*   READ AND DISPLAY RECORDS
READLOOP LA   R1,WORKAREA
         GET  FILEDCB,WORKAREA
         LTR  R15,R15
         BZ   PROCESS_RECORD   
         B    EOF1              
*
PROCESS_RECORD
         MVC 0(80,R1),WORKAREA
         WTO  MF=(E,(R1))
         B    READLOOP
*
EOF1     CLOSE (FILEDCB)
         L   R13,8(R13)
         LM R14,R12,12(R13)
         BR   R14
         LTORG
         END
```

This code demonstrates a basic file operation: writing ten 80-byte records and reading them back. Here, the `FILEDCB` is statically defined. Suppose now that another job or utility modified the output file's `BLKSIZE` before the 'READ' section of the code is executed. If the second portion of the program is re-executed, the `FILEDCB` will still reflect the original `BLKSIZE` value, which will no longer agree with the actual physical characteristics of the data set, making proper reads and EOF detection unreliable.

To further illustrate, consider how dynamically allocated output files impact behavior. The following modified example creates an output file dynamically:

**Example 2: Dynamic Allocation and Potential Issue**

```assembler
        TITLE 'FILEIO EXAMPLE 2'
        PRINT NOGEN
        SYSSTATE AMODE=31
*
*     DEFINE WORK AREAS
        DSECT ,
SAVEAREA DS   18F         Savearea
*
WORKAREA DS  CL100       Work buffer
FILEDCB  DCB   DDNAME=OUTPUT,DSORG=PS,MACRF=PM,       X
              EODAD=EOF2
*
        CSECT ,
        USING *,R13
        ST   R13,SAVEAREA+4
        LR    R12,R13
        ST   R12,8(R13)
        LR   R13,SP
*
* DYNAMIC ALLOCATION VIA SVC 99
        LA R1,PARM
        SVC 99
        LR R0,R15
        BZ ALLOC_OK
        WTO MF=(E,(R1)),TXT='ALLOCATION FAILED'
        B  ABORT
ALLOC_OK
         OPEN (FILEDCB,OUTPUT)
*
* WRITE SAMPLE RECORDS
         LA   R1,WORKAREA
         MVI  0(R1),C'B'      
         LA  R2,10
LOOP2    STC R2,1(R1)
         PUT  FILEDCB,WORKAREA
         BCT R2,LOOP2
*
* CLOSE OUTPUT
         CLOSE (FILEDCB)
*
         OPEN (FILEDCB,INPUT)
*   READ AND DISPLAY RECORDS
READLOOP2 LA   R1,WORKAREA
         GET  FILEDCB,WORKAREA
         LTR  R15,R15
         BZ   PROCESS_RECORD2   
         B    EOF2           
*
PROCESS_RECORD2
         MVC 0(80,R1),WORKAREA
         WTO  MF=(E,(R1))
         B    READLOOP2
*
EOF2     CLOSE (FILEDCB)
ABORT     L   R13,8(R13)
         LM R14,R12,12(R13)
         BR   R14
*
*     ALLOCATION PARAMETERS
PARM     DC   X'00000000'         Parameter length
         DC   X'01000000'         Function - Allocate
         DC   A(DDNAME),         DDname address
         DC   CL8'OUTPUT  '        DDNAME
         DC   X'04000000'         DATACLAS Attribute length
         DC   CL8'         '   DATACLAS
         DC   X'01000000'        SPACE Attribute Length
         DC    A(SPACEPARM),      SPACE parameter address
SPACEPARM DC  X'00000000',X'00000000',X'00000000'
         DC  X'0000000A',X'00000000'
         DC  X'00000014',X'00000000',X'00000000',X'00000000'
*
         LTORG
         END
```

In this second example, the file is dynamically allocated using SVC 99. The data set is created, ten records are written, and the file closed. After this the file is opened again to read, as with Example 1. If this is run multiple times or if another job modifies the file attributes through batch JCL between executions of the second portion, the issues are the same as Example 1, and the program can loop erroneously.

To illustrate a scenario that specifically changes attributes dynamically, I'll present a third code segment. This segment demonstrates a file created, closed, then modified using JCL and reopened by another assembly process:

**Example 3: Attribute Change via JCL**

(Assuming the initial assembly program is the second example compiled and executed, called 'PROGRAM1')
  
**JCL for initial run (JOB1):**

```jcl
//JOB1     JOB  ...
//* EXECUTE ASSEMBLER PROGRAM 1
//STEP1     EXEC PGM=PROGRAM1
//OUTPUT    DD DSN=MY.OUTPUT.FILE,DISP=(NEW,CATLG,DELETE),
//             UNIT=SYSDA,SPACE=(TRK,(1,1)),
//             RECFM=FB,LRECL=80,BLKSIZE=2792
```

Now, after `PROGRAM1` has executed, the following JCL is submitted, modifying the file before `PROGRAM1` is rerun.
 
**JCL for modification (JOB2):**

```jcl
//JOB2     JOB  ...
//MODFILE  EXEC PGM=IEFBR14
//OUTPUT   DD DSN=MY.OUTPUT.FILE,DISP=OLD,
//          RECFM=VB,LRECL=100,BLKSIZE=6233
```

And finally the 're-run' of the assembler program.

**JCL for re-run (JOB3):**

```jcl
//JOB3     JOB  ...
//*EXECUTE ASSEMBLER PROGRAM 1 AGAIN
//STEP1     EXEC PGM=PROGRAM1
//OUTPUT    DD DSN=MY.OUTPUT.FILE,DISP=SHR
```

In this scenario, `JOB2` alters the file attributes of `MY.OUTPUT.FILE` (RECFM, LRECL, BLKSIZE). When `PROGRAM1` is executed again in `JOB3`, it may still retain a cached version of the file's original attributes. The OPEN macro used by the program will not necessarily cause a re-read of the attributes from disk, which means that the read loop can misinterpret the end-of-file condition, triggering infinite looping.

To prevent these looping issues, an assembler program should:

1.  **Explicitly manage File Attributes:** If the application requires specific file attributes, ensure that these are consistently set through JCL allocation or dynamic allocation via SVC 99, and do not assume they remain static.
2.  **Re-read File Attributes:** If the output file is potentially manipulated outside the program, include logic to explicitly re-read file attributes using the GETFILE macro or associated control block fields before crucial operations like reads or writes.
3.  **Careful Error Handling:** Thoroughly check return codes from I/O operations to catch errors that indicate incorrect end-of-file conditions. This can include utilizing the `SYNADAF` macro to capture specific error conditions and diagnose problems early.
4.  **Dynamic Allocations with Caution:** When using SVC 99, establish robust parameter handling to prevent invalid specifications or conflicts, especially if the program is executed multiple times.
5. **Use File Control Blocks Directly:** The program can directly access the file control block, which is a control block that contains meta-data about a file. The program should access this for up-to-date information about file metadata, instead of relying on a cached or previously read value of the file metadata. 

For deeper study on these concepts, I would recommend consulting the following IBM publications:

*   *z/OS MVS Programming: Assembler Services Reference* (for details on the GET, PUT, OPEN, CLOSE, and related macros, as well as system service calls, such as SVC 99).
*   *z/OS MVS JCL Reference* (to better understand how file attributes are controlled through JCL).
*   *z/OS DFSMS Macro Instructions for Data Sets* (for detail on the use of Data Control Blocks (DCBs) and Data Set Control Blocks (DSCBs)).

These resources provide in-depth details on the underlying mechanisms that can influence file I/O behavior, thus helping one build robust z/OS assembler programs. The interaction between file attributes, application assumptions, and system operations requires close attention to detail to prevent unforeseen issues like erratic looping.
