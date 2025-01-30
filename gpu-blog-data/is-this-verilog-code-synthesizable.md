---
title: "Is this Verilog code synthesizable?"
date: "2025-01-30"
id: "is-this-verilog-code-synthesizable"
---
The synthesizability of Verilog code hinges critically on its adherence to the specific constraints and capabilities of the target FPGA architecture.  My experience working on high-performance network processors at Xylos Corp. highlighted this frequently.  While seemingly straightforward, many constructs that function correctly in simulation fail to map effectively onto physical hardware during synthesis.  Therefore, a blanket "yes" or "no" answer to whether *any* Verilog code is synthesizable is inaccurate.  A detailed analysis is always necessary.

Let's consider this question within the context of several common scenarios that often lead to synthesizability issues.  I've observed these numerous times throughout my career.

**1.  Blocking vs. Non-blocking Assignments:** The distinction between `<=` (non-blocking) and `=` (blocking) assignments is paramount.  Blocking assignments execute sequentially within a single `always` block, while non-blocking assignments schedule their updates concurrently.  Improper use can lead to unintended results in hardware, preventing successful synthesis.  For sequential logic, non-blocking assignments are generally preferred.  Misusing blocking assignments within combinatorial logic can easily lead to unintended latches which are often undesired in design, and will cause synthesis warnings or errors, depending on the tool.

**2.  Implicit and Explicit Latches:**  Unintentional latches are a frequent source of synthesis problems.  They occur when a variable within a `always` block is assigned under some conditions but left unassigned under others.  Synthesis tools often infer latches in such situations to maintain state, leading to unpredictable behavior and potentially higher resource usage compared to what would be the case with explicit latches or state-holding logic elements.  Properly defining all possible assignment conditions, even if it means explicitly assigning a default value, prevents this issue.

**3.  Data Types and Range Constraints:**  Using incorrect data types or omitting range constraints can hinder synthesizability. The synthesis tool needs precise information about data width to generate the correct hardware elements.  Furthermore, using unconstrained arrays or vectors can lead to unpredictable hardware instantiation.  Precise data type declarations, particularly with bit ranges specified for each signal, are crucial for effective synthesis.

**4.  System Tasks and Functions:**  System tasks and functions, such as `$display`, `$monitor`, and `$finish`, are essential for simulation but are not synthesizable. They're purely for debugging and simulation purposes.  Including them in code intended for synthesis will result in errors.  These should be rigorously excluded from any portion of the code intended for hardware implementation.  Conditional compilation directives (``ifdef`, `ifndef`, `endif`) are frequently used to manage this.

**5.  Algorithmic constructs in synthesis:** Algorithmic constructs such as loops or complex conditional statements must be carefully considered.  The synthesis tool aims to translate these into efficient hardware. Unbounded loops, or loops that have a number of iterations dependent on unpredictable runtime values, can cause synthesis difficulties, sometimes causing synthesis to fail completely.  In general, synthesizable Verilog should be primarily expressed through concurrent assignments and synchronous hardware components.  Algorithms that are fundamentally iterative should be transformed into more directly hardware-compatible constructs.



**Code Examples and Commentary:**

**Example 1: Non-synthesizable code (due to improper use of blocking assignments):**

```verilog
always @(posedge clk) begin
  a = b;
  b = a + 1;
end
```

This code, while seemingly correct in simulation, uses blocking assignments in a sequential always block which will result in incorrect values.  The intended behaviour likely involves using non-blocking assignments:


```verilog
always @(posedge clk) begin
  a <= b;
  b <= a + 1;
end
```
This revised code uses non-blocking assignments, ensuring concurrent updates and correct behavior in hardware.


**Example 2: Synthesizable code (simple counter):**

```verilog
module counter (
  input clk,
  input rst,
  output reg [7:0] count
);
  always @(posedge clk) begin
    if (rst)
      count <= 8'b0;
    else
      count <= count + 1;
  end
endmodule
```

This is a straightforward synthesizable counter. It utilizes non-blocking assignments, has a clear reset condition, and defines the output's data type precisely. It clearly demonstrates the use of a synchronous process within an always block. This code will synthesize reliably.


**Example 3:  Illustrating implicit latches:**

```verilog
module latch_example (
  input a,
  input b,
  output reg y
);
  always @(a or b) begin
    if (a)
      y = 1'b1;
  end
endmodule
```

This code will likely synthesize with an implicit latch on 'y' due to the incomplete assignment. When 'a' is low, the value of 'y' is not explicitly assigned and a latch will be inferred to store the previous value of 'y'.  To avoid this, the code should be rewritten to provide a complete assignment condition:

```verilog
module latch_example_fixed (
  input a,
  input b,
  output reg y
);
  always @(a or b) begin
    if (a)
      y = 1'b1;
    else
      y = 1'b0; // Explicitly assign a value when 'a' is 0
  end
endmodule
```
This version explicitly assigns 'y' under all conditions, preventing the synthesis tool from inferring an unintended latch.


**Resource Recommendations:**

For further study, I would recommend consulting the official language reference manual for your specific Verilog version, a comprehensive text on digital design with Verilog HDL, and a guide focused on Verilog synthesis and optimization techniques.  Understanding the nuances of different FPGA architectures and their specific synthesis tools is also highly valuable for practical application.  Focus on understanding the distinction between behavioral modeling and RTL modeling.  The differences and their practical implications are key to writing synthesizable Verilog.
