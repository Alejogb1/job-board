---
title: "How can I resolve REXX program RC passing issues from a MACRO?"
date: "2025-01-30"
id: "how-can-i-resolve-rexx-program-rc-passing"
---
The root cause of REXX program return code (RC) passing inconsistencies from within a macro often stems from a misunderstanding of how the underlying z/OS environment handles program invocation and return code propagation.  Specifically, the mechanism by which a REXX program, launched from a macro, communicates its completion status back to the calling macro isn't always straightforward and relies on correctly interpreting the `ADDRESS` instruction within the REXX script and the subsequent handling of the RC within the invoking macro.  In my experience troubleshooting similar issues across numerous z/OS batch jobs and TSO/E sessions, consistent adherence to certain practices significantly minimizes these issues.

**1.  Clear Explanation of the Problem and Solution**

The problem arises from the ambiguity in how REXX handles the return codes.  A REXX program doesn't inherently return an RC in the same way a compiled program does. Instead, the final value of the special variable `rc` within the REXX program is used as a proxy for the return code.  If a REXX program is executed using the `ADDRESS TSO` or `ADDRESS ISPEXEC` instructions,  the RC passed back to the caller is influenced by whether the called REXX program explicitly sets the `rc` variable and the handling of any potential errors within the called program.  Failure to explicitly set `rc` or improper error handling can lead to unpredictable RC values passed back to the invoking macro, often resulting in seemingly random failures or incorrect logic flow in the parent macro.

The solution lies in meticulously managing the `rc` variable within the REXX program and using appropriate error-handling mechanisms.  The macro should also explicitly check the `rc` value returned from the REXX program and act accordingly. Furthermore, the `ADDRESS` instruction plays a crucial role – choosing the incorrect target can lead to unexpected RCs. Using `ADDRESS` with `TSO` or `ISPEXEC` correctly and consistently contributes to reliable code behavior.

**2. Code Examples with Commentary**

**Example 1: Correct REXX Program and Macro Interaction**

This example demonstrates a well-structured REXX program that correctly sets its `rc` based on successful or unsuccessful execution and a macro that handles the RC appropriately:

```rexx
/* REXX Program: myREXXProgram.rexx */
arg inputVar
"Some processing using inputVar"
if condition then do
  rc = 0
  say "Processing successful"
end
else do
  rc = 16
  say "Processing failed"
end
exit rc

/* Macro: myMacro.jcl */
//MYMACRO  EXEC PGM=IKJEFT01
//SYSTSIN  DD *
  ADDRESS TSO "EXEC 'myREXXProgram' inputData"
  rc = %SYSPROC(RETURN)
  if rc = 0 then
    do
      say "REXX program executed successfully"
    end
  else
    do
      say "REXX program failed with RC:" rc
      abend rc
    end
/*
Explanation: The REXX program explicitly sets rc to 0 for success and 16 for failure. The macro uses ADDRESS TSO to invoke the REXX program.  The %SYSPROC(RETURN) built-in function captures the REXX program's RC. Subsequently, the macro's logic branches based on the returned RC.  Note that the abend statement in the macro can be replaced by alternative error handling mechanisms based on the macro's function.
*/
```

**Example 2: Incorrect Handling of Errors in REXX**

This example shows an incorrect approach where error handling is missing in the REXX program, leading to unpredictable RCs:

```rexx
/* REXX Program: myREXXProgram_Err.rexx */
arg inputVar
"Some processing using inputVar that might fail"
/* No explicit rc setting if an error occurs */
exit
/*
Explanation:  This program lacks explicit `rc` setting.  If "Some processing..." encounters an error (e.g., file I/O failure), the returned `rc` will be determined by the underlying system error, leading to inconsistencies.
*/
```

**Example 3: Using Incorrect ADDRESS Target**

This demonstrates the importance of selecting the appropriate `ADDRESS` target:

```rexx
/* REXX Program: myREXXProgram_Addr.rexx */
say "This program will always return 0"
exit 0

/* Macro: myMacro_Addr.jcl */
//MYMACRO  EXEC PGM=IKJEFT01
//SYSTSIN  DD *
  ADDRESS COMMAND "EXEC 'myREXXProgram_Addr'" /* Incorrect use of ADDRESS */
  rc = %SYSPROC(RETURN)
  if rc = 0 then
    do
      say "REXX program executed successfully"
    end
  else
    do
      say "REXX program failed with RC:" rc
    end
/*
Explanation: Although the REXX program correctly sets rc, using ADDRESS COMMAND might not propagate the rc correctly. This could lead to the macro receiving unexpected RC values or the macro execution failing for unrelated reasons depending on the command being executed.  Using ADDRESS TSO is generally preferred for REXX execution.
*/
```


**3. Resource Recommendations**

For in-depth understanding of REXX programming under z/OS, I recommend consulting the official z/OS documentation related to REXX language syntax and execution.  Studying examples of well-structured REXX programs and macros will aid in understanding best practices for RC handling.  Furthermore, the IBM publications concerning z/OS JCL and the various system services leveraged within macros, particularly those related to program invocation and return code interpretation, will prove extremely valuable. Examining existing z/OS codebases within your organization (with proper authorization) for similar processes provides invaluable insights into practical implementations and error handling strategies.  Finally, actively using a z/OS debugging tool for thorough testing and analysis of the interactions between the macro and the REXX program is crucial.  These steps will help you accurately identify and resolve unexpected behavior related to RC propagation.
