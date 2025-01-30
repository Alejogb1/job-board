---
title: "What causes Quartus II errors in my SystemVerilog code?"
date: "2025-01-30"
id: "what-causes-quartus-ii-errors-in-my-systemverilog"
---
SystemVerilog code targeting Altera (now Intel) FPGAs through Quartus II often encounters errors arising from a mismatch between the high-level abstraction of the language and the underlying hardware architecture, specifically how the synthesis engine interprets that code. These errors are rarely due to outright syntactic mistakes, which a compiler would typically catch. Instead, they tend to stem from violating synthesis constraints or unintended hardware implications embedded within the SystemVerilog description. My experience troubleshooting these issues over several years across multiple projects suggests these errors frequently fall into a few specific categories.

First, many Quartus II errors originate from misunderstandings regarding clock domain crossings (CDCs). SystemVerilog allows the definition of multiple clock domains, which is necessary for designing complex systems. However, naively transferring data between these domains without proper synchronization mechanisms leads to metastability and unpredictable behavior. Consider the following scenario I encountered:

```systemverilog
module bad_cdc (
    input  logic clk_a,
    input  logic clk_b,
    input  logic [7:0] data_in,
    output logic [7:0] data_out
);

  logic [7:0]  sync_data;

  always_ff @(posedge clk_a)
    sync_data <= data_in;

  always_ff @(posedge clk_b)
    data_out <= sync_data;

endmodule
```

This code defines two flip-flops; the first is clocked by `clk_a` and the second by `clk_b`. While it may appear logically sound, the asynchronous relationship between `clk_a` and `clk_b` means the data `sync_data` may change near the rising edge of `clk_b`. This can cause the second flip-flop to enter a metastable state, where it oscillates between high and low before eventually settling to a stable state, but to an unpredictable level. This ultimately violates the Quartus timing constraints and results in a 'setup/hold' time error during place and route, often manifesting as a timing failure in the fitter stage. Such scenarios are often difficult to debug initially as the behaviour might be intermittent, depending on the precise timings of each clock edge. The Quartus II error reports, in this case, would highlight timing violations, often without directly indicating the CDC as the root cause, requiring the designer to manually trace data paths.

Second, improper use or incorrect inferences of latches is a frequent source of Quartus II errors, usually arising from incomplete sensitivity lists or conditional assignments. SystemVerilog's flexibility can sometimes lead to unintended latch creation by the synthesis tool. Consider:

```systemverilog
module latch_inference (
  input   logic [3:0] sel,
  input   logic [7:0] data_a,
  input   logic [7:0] data_b,
  output  logic [7:0] out
);

  always_comb
    if (sel[0])
      out = data_a;
    else if (sel[1])
      out = data_b;
    // No default case for when sel[0] and sel[1] are both 0

endmodule
```

In this module, `out` is only assigned a value under specific conditions (`sel[0]` or `sel[1]` being true). When both `sel[0]` and `sel[1]` are false, the previous value of `out` is held, which results in the synthesis engine inferring a latch. Latches, while valid in certain circuit designs, are frequently undesirable in FPGA implementations as they exhibit poor timing characteristics and can be vulnerable to glitches, contributing to timing errors during the place and route process, and increasing the chances of setup/hold violations. The fitter might also introduce additional routing overhead to accommodate latches, leading to resource utilization errors. Quartus II errors concerning latches tend to refer to inferred latch elements and often accompany fitter errors relating to timing constraints. In my experience, a design rule check (DRC) will identify this prior to physical implementation. The resolution typically involves either adding a default assignment to `out` within all possible execution paths of the conditional logic, or utilizing a synchronous coding style.

Third, resource exhaustion, especially on smaller FPGAs, can cause Quartus II to report errors even when there are no logical flaws in the code. This commonly occurs with over-reliance on certain primitives such as multipliers or excessive instantiation of complex IP cores. I encountered this while implementing a high-speed FIR filter on a small FPGA board with limited resources.

```systemverilog
module resource_exhaustion (
  input   logic clk,
  input   logic signed [7:0] in_data,
  output  logic signed [15:0] out_data
);

  logic signed [7:0] coefs[15:0]; //16 coefficients

  logic signed [15:0] mult_results [15:0];
  logic signed [31:0] acc_result;

  always_ff @(posedge clk) begin
    for (int i = 0; i < 16; i++) begin
      mult_results[i] <= in_data * coefs[i];
    end

    acc_result <= '0;
    for (int j = 0; j < 16; j++) begin
      acc_result <= acc_result + mult_results[j];
    end

    out_data <= acc_result[23:8]; // Arbitrary range for example.
  end


endmodule
```

This design uses a series of multiplies and an adder tree in a single clock cycle. Although this is syntactically and logically correct, Quartus II will report errors if the number of DSP blocks and routing resources cannot accommodate this level of parallelism within the target FPGA. This type of error, while not directly related to the code’s correctness in a logical sense, indicates an imbalance between design requirements and hardware constraints. The errors are typically manifested as fitter stage failure, resource exhaustion warnings, or difficulty placing and routing the design. Resolution requires re-architecting the design, potentially by time-multiplexing the operations, using a pipelined architecture, or utilizing a different algorithm to achieve the desired output. The error messages within the Quartus II reports provide information about the resource type and quantity over-utilized.

For further understanding and resolution, I would suggest the following resources, each offering valuable insights and specific guidance: The “Intel FPGA Documentation” provides comprehensive information about supported primitives, synthesis guidelines, and hardware architectures. Similarly, "SystemVerilog for Synthesis" by various authors, offer practical insights into synthesizable coding styles and avoid common pitfalls. And the 'Intel FPGA Knowledge Base' provides targeted solutions for error messages and is highly valuable in diagnosing implementation specific issues. Working with these resources, and having a solid foundational understanding of both SystemVerilog and FPGA architecture is key to avoiding common Quartus II synthesis, fitting and timing issues.
