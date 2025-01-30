---
title: "Why is Verilog code functional in ISIM (Xilinx 14.2) but not on Spartan-6 hardware?"
date: "2025-01-30"
id: "why-is-verilog-code-functional-in-isim-xilinx"
---
The discrepancy between Verilog code functioning correctly in ISIM and failing on Spartan-6 hardware frequently stems from unsynthesizable constructs employed within the design.  My experience debugging similar issues over the past decade, primarily involving complex state machines and memory-mapped interfaces in Xilinx FPGAs, indicates that this is a pervasive problem.  While ISIM provides a cycle-accurate simulation environment, it lacks the constraints and limitations inherent in actual hardware implementation.  Consequently, code that simulates flawlessly might contain elements incompatible with the target FPGA's architecture.

**1. Clear Explanation:**

The core problem lies in the difference between simulation and synthesis. ISIM, a behavioral simulator, executes the Verilog code line by line, largely ignoring hardware-specific restrictions. Synthesis, however, translates the Verilog into a netlist describing the hardware's configuration. This process discards constructs that are not mappable to the FPGA's logic elements, memory blocks, and interconnect.  Common culprits include:

* **Improper use of `always` blocks:**  `always` blocks with implicit event control (e.g., lacking a precise `@` sensitivity list) can lead to unpredictable behavior in synthesis.  The synthesizer might infer different hardware structures than anticipated, resulting in functional divergence between simulation and implementation.

* **Blocking vs. Non-blocking assignments:**  Mixing blocking (`=`) and non-blocking (`<=`) assignments within the same `always` block can produce unexpected results in hardware.  Blocking assignments execute sequentially, while non-blocking assignments execute concurrently.  This difference impacts the order of signal updates, leading to discrepancies between the simulated and synthesized behavior.

* **Inferred latches:**  Verilog synthesizers may infer latches if the output of an `always` block is not fully assigned under all possible conditions.  Latches are less predictable and less efficient than flip-flops, and their presence can significantly impact the timing and functionality of the hardware implementation.

* **Unsupported data types or operations:**  While ISIM might handle complex data types or operations, the target FPGA may not directly support them.  This often necessitates manually converting the data structures into FPGA-compatible representations (e.g., using arrays of bits instead of arbitrarily sized integers).

* **Timing constraints and clock domains:**  ISIM typically ignores timing constraints.  However, on hardware, timing closure is critical.  Asynchronous signals crossing clock domains can cause metastability issues, leading to unpredictable behavior.  Ignoring these issues in the design leads to functionality discrepancies between simulation and hardware.


**2. Code Examples and Commentary:**

**Example 1: Implicit Sensitivity List in `always` Block:**

```verilog
module faulty_counter (input clk, input rst, output reg [7:0] count);
  always @(posedge clk) begin
    if (rst)
      count <= 8'b0;
    else
      count <= count + 1;
  end
endmodule
```

This code might work correctly in ISIM. However, the implicit sensitivity list (`@posedge clk`) is problematic.  The synthesizer might infer a latch for `rst`, leading to unexpected behavior if the reset signal changes asynchronously with the clock.  A correct implementation would explicitly specify the sensitivity list:

```verilog
module corrected_counter (input clk, input rst, output reg [7:0] count);
  always @(posedge clk or posedge rst) begin
    if (rst)
      count <= 8'b0;
    else
      count <= count + 1;
  end
endmodule
```

**Example 2: Mixing Blocking and Non-blocking Assignments:**

```verilog
module mixed_assignment (input clk, input a, input b, output reg c);
  always @(posedge clk) begin
    c = a; // Blocking assignment
    c <= b; // Non-blocking assignment
  end
endmodule
```

In this example, the blocking assignment to `c` from `a` will overwrite the subsequent non-blocking assignment from `b`.  The synthesized hardware will reflect this overwrite, possibly leading to unexpected behavior.  Consistent use of non-blocking assignments is generally recommended for sequential logic:

```verilog
module corrected_assignment (input clk, input a, input b, output reg c);
  always @(posedge clk) begin
    c <= a; // Non-blocking assignment
    c <= b; // Non-blocking assignment - last assignment wins
  end
endmodule
```


**Example 3: Unassigned Outputs Leading to Latch Inference:**

```verilog
module latch_example (input clk, input enable, output reg out);
  always @(posedge clk) begin
    if (enable)
      out <= 1'b1;
  end
endmodule
```

This design lacks an assignment for `out` when `enable` is low.  The synthesizer will likely infer a latch to maintain the previous value of `out`, potentially resulting in unwanted behavior.  A safe approach involves explicitly assigning a value to `out` under all conditions:

```verilog
module corrected_latch (input clk, input enable, output reg out);
  always @(posedge clk) begin
    if (enable)
      out <= 1'b1;
    else
      out <= 1'b0; // Explicit assignment when enable is low
  end
endmodule
```


**3. Resource Recommendations:**

* Xilinx Synthesis and Simulation documentation:  This provides comprehensive guidance on synthesis constraints, supported data types, and best practices for Verilog coding for Xilinx FPGAs.

* Verilog HDL reference manual:  A thorough understanding of Verilog syntax and semantics is essential for preventing synthesis-related issues.

* Advanced FPGA design textbooks:  These delve into the intricacies of FPGA architectures, timing constraints, and clock domain crossing, crucial for understanding the limitations of hardware implementation.


In conclusion, the discrepancy between ISIM simulation and Spartan-6 hardware behavior is primarily attributable to the differences in how these environments handle Verilog code. Careful adherence to synthesis-friendly coding practices, a strong grasp of Verilog fundamentals, and a thorough understanding of FPGA architecture are essential for bridging this gap and creating reliable, functional hardware designs.  Ignoring these factors leads to considerable debugging challenges, as I've encountered repeatedly throughout my career.
