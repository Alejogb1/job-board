---
title: "Why is my Quartus II design compiling successfully but showing no logic utilization?"
date: "2025-01-30"
id: "why-is-my-quartus-ii-design-compiling-successfully"
---
The absence of logic utilization in a Quartus II design despite a successful compilation typically indicates a problem with how the design is being synthesized, not a compiler error.  My experience troubleshooting similar issues over the years points to several common root causes, primarily centered around the misinterpretation of design constraints or the unintended optimization of logic away by the synthesizer.

**1.  Unconstrained Signals and Optimization:**

The most frequent cause I've encountered is the lack of explicit constraints on signals, allowing Quartus II's synthesizer to perform aggressive optimizations that effectively eliminate logic elements.  The synthesizer prioritizes resource efficiency; if it deems a signal's logic functionally redundant, it will optimize it out. This is especially problematic with combinational logic where the output is only dependent on the present inputs and not previous states.  If such logic's output isn't explicitly used in the design, Quartus II might consider it dead code and remove it.  Furthermore, improperly specified register assignments can lead to registers being optimized away if they lack functionality in the post-synthesis netlist.


**2.  Incorrect Clocking Strategy:**

Another common issue stems from the clocking structure. If the design uses clocks that are not properly defined or connected to flip-flops, registers might not be correctly inferred.  A missing clock assignment, an incorrect clock frequency constraint, or a faulty clock distribution network can all lead to seemingly successful compilations, yet produce no observable logic usage. This is because Quartus II might be successful in parsing the HDL, but if the timing constraints are inadequate or impossible to meet, the synthesizer will struggle to map the design to actual hardware, resulting in zero utilization.



**3.  Hierarchical Design Issues:**

In larger, hierarchical designs, problems can arise from improper instantiation of modules or missing connections between them.  One module might compile independently without errors, but if its output is never connected or utilized by the parent module, its logic will not be included in the final implementation. This is frequently overlooked; during top-level synthesis, if the higher-level module doesn't actively use the outputs from a lower-level module, Quartus II will eliminate the lower-level logic since it deems the result irrelevant.


**Code Examples and Commentary:**

**Example 1: Unconstrained Combinational Logic**

```verilog
module unconstrained_logic (input a, b, output c);
  assign c = a & b; //Simple AND gate, output c not used elsewhere
endmodule

module top_module;
  wire a, b;
  wire c;
  unconstrained_logic uut (a, b, c); // Instance, but c is unconnected
  // ...rest of the design...
endmodule
```

Here, the `c` signal is not used anywhere in the `top_module`. The synthesizer correctly infers the AND gate in `unconstrained_logic`, but since `c` is not utilized, Quartus II optimizes it away, resulting in zero utilization.  To fix this, ensure all signals are connected and used within the design hierarchy.


**Example 2: Incorrect Clock Assignment**

```verilog
module faulty_clocking (input clk, input rst, output reg out);
  always @(posedge clk) begin // No reset assignment
    if (rst) out <= 0;
    else out <= ~out;
  end
endmodule

module top_module;
  wire clk;
  wire rst;
  wire out;
  faulty_clocking uut (clk, rst, out);
  // ...rest of the design...
endmodule
```

In this example, the reset signal `rst` is improperly used in the `always` block. Additionally, the clock signal `clk`  might not be properly defined or constrained in the top module, leading to incorrect register synthesis or no register synthesis at all.  To correct this, ensure proper reset handling and correct clock signal assignment with defined frequency constraints in the top-level design file.  Also verify the clock source is correctly defined and assigned.


**Example 3:  Hierarchical Design Error**

```verilog
module sub_module (input a, output b);
  assign b = ~a;
endmodule

module top_module;
  wire a;
  //wire b;  This line is commented out, causing the issue.
  sub_module uut (a, b);
endmodule
```

The `top_module` instantiates `sub_module` but the output `b` is not declared as a wire.  This prevents any connection between the sub-module's output and the top-level design.  Quartus II correctly synthesizes `sub_module`, but because its output is unconnected, the logic is not used and therefore isn't reflected in the resource utilization report.  Adding `wire b;` will resolve the problem.


**Resource Recommendations:**

The Quartus II documentation, particularly the sections on synthesis, constraints, and timing analysis, provides invaluable information.  Familiarity with Verilog or VHDL coding standards is crucial.  Understanding the concepts of  hierarchical design, clock domain crossing, and register inference is equally important. The Quartus II software includes numerous tutorials and examples to aid in the learning process. Finally, carefully reviewing the synthesis reports, especially the "Logic Utilization" section and "Timing Analysis" report, is essential for pinpointing the root cause of any compilation issues.
