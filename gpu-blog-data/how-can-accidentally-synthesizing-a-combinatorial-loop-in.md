---
title: "How can accidentally synthesizing a combinatorial loop in a sequential always block be avoided?"
date: "2025-01-30"
id: "how-can-accidentally-synthesizing-a-combinatorial-loop-in"
---
Accidental synthesis of combinatorial loops in sequential always blocks is a frequent source of unexpected behavior and synthesis failures in Verilog and SystemVerilog designs.  My experience debugging thousands of lines of RTL, particularly in high-speed data processing systems, has shown that the root cause often lies in unintended feedback paths created by improperly structured assignments within the always block.  Specifically, the problem arises when a signal's next-state value depends directly or indirectly on its current value within the same clock cycle.  This creates a cyclic dependency that the synthesis tool cannot resolve, resulting in unpredictable simulation behavior and likely synthesis errors flagged as latch generation or combinatorial loops.


**1.  Clear Explanation:**

A sequential always block, typically identified by its sensitivity list containing a clock signal and potentially asynchronous reset, is intended to model synchronous logic.  The code within describes the next-state values of signals based on their current state and input values.  The crucial aspect is that these next-state assignments are only effective *at the clock edge*.  A combinatorial loop occurs when a signal's next-state is calculated using its current value *without* waiting for the next clock edge.  This can happen subtly, often through unintended dependencies between signals within the same always block.  The synthesis tool interprets this as a requirement to continuously evaluate the signal's value, leading to an infinite loop during synthesis.  This often manifests as a latch inference (the synthesizer attempts to resolve the undefined behavior by creating a latch), or directly as a combinatorial loop error.  To prevent this, ensure that all signals whose next-state is defined within the always block have their new values derived solely from signals that are either:

*   Inputs to the block.
*   Signals assigned in previous clock cycles (i.e., their values are stable at the clock edge).
*   Signals derived through completely combinational logic outside the sequential always block, provided their dependency graph does not create a cycle.


**2. Code Examples with Commentary:**

**Example 1: Incorrect – Combinatorial Loop**

```verilog
always @(posedge clk) begin
  if (reset) begin
    count <= 0;
  end else begin
    count <= count + 1;  // Incorrect: count's next state depends on its current state within the same clock cycle.
  end
end
```

This example appears correct at first glance. However, the assignment `count <= count + 1;` creates the combinatorial loop.  The synthesizer attempts to resolve `count`'s current value to determine its next value, creating the cycle. The solution is to use an intermediate variable.


**Example 2: Correct – Intermediate Variable**

```verilog
always @(posedge clk) begin
  if (reset) begin
    count <= 0;
    next_count <= 0;
  end else begin
    next_count <= count + 1; // Calculate next_count based on current count
    count <= next_count;     // Assign next_count to count at the next clock edge
  end
end
```

This corrected version introduces `next_count`. The calculation of the next value occurs independently of `count`'s current state at the clock edge. At the next clock edge, `next_count`'s value is safely assigned to `count`. This eliminates the direct dependency during the current clock cycle, breaking the combinatorial loop.

**Example 3: Incorrect – Unintended Feedback through Combinational Logic**

```systemverilog
always_ff @(posedge clk) begin
  if (reset)
    data_out <= 0;
  else
    data_out <= process_data(data_in, data_out); // Incorrect: Feedback through combinational function
end

function automatic logic [7:0] process_data;
  input logic [7:0] a, b;
  process_data = a ^ b; //Example combinational logic; could be far more complex
endfunction
```

This example demonstrates a more subtle loop. The `process_data` function uses `data_out` as an input, creating feedback within the same clock cycle.  Even though `process_data` is combinational, the feedback within the always block creates a cyclic dependency.


**3. Resource Recommendations:**

*   Refer to your chosen synthesis tool's documentation on latch inference and combinatorial loop detection.  Thorough understanding of the synthesis process is paramount.
*   Consult advanced Verilog or SystemVerilog textbooks covering RTL design and synthesis best practices. Focus on sections detailing sequential logic design and common pitfalls.  Pay close attention to examples demonstrating correct and incorrect usage of always blocks and signal assignments.
*   Explore design methodologies, such as formal verification techniques, which can help identify such subtle errors during design review. These formal methods can systematically check for the absence of combinatorial loops.  Using a static timing analyzer is also important after synthesis.


Throughout my career, I've observed that a systematic approach to RTL coding, coupled with rigorous code review and appropriate usage of synthesis tools, is the most effective strategy for preventing these problems.  Careful consideration of signal dependencies and the use of intermediate variables where necessary are key to building robust and synthesizable designs.  The avoidance of self-referential assignments within the same clock cycle within a sequential always block is the fundamental principle to keep in mind.  The examples above illustrate common mistakes and demonstrate how to implement the correct solutions to avoid accidental combinatorial loop creation.  A well-defined and understood design methodology is far more effective in the long run than relying on post-synthesis error messages to diagnose these problems.
