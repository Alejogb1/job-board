---
title: "Why is XST optimizing away my registers, and how can I prevent this?"
date: "2025-01-30"
id: "why-is-xst-optimizing-away-my-registers-and"
---
I’ve spent the last decade or so working with Xilinx FPGAs, primarily in high-throughput digital signal processing applications. Register optimization is a frequent hurdle in those designs, and I've directly encountered the issue of XST—the Xilinx Synthesis Tool—aggressively removing registers that I thought were necessary. This behavior, while often beneficial for resource utilization, can lead to unexpected functionality changes, especially during debugging. My initial reaction, more often than not, was to suspect a coding error on my part. Over time, however, I’ve learned the intricacies of how XST interprets HDL code and the precise techniques required to retain specific registers.

The core reason XST optimizes away registers boils down to its primary directive: minimize resource consumption and maximize performance. From the tool's perspective, any register that does not contribute to the final output, or is considered redundant, is a candidate for removal. This is achieved through a process called register balancing, where the tool analyzes the data flow. If the input to a register is identical to the register's output, it determines that this register serves no logical purpose and eliminates it. The logic is not wrong, but the application of that logic can result in unexpected outcomes when the developer intends to hold intermediate states for other purposes like internal timing or debugging. It is important to understand that XST optimizes based solely on logical requirements of the described circuit, not on arbitrary coding structures.

The root cause of this optimization is often the absence of a clear, externally observable output that depends on the register's state. A common scenario involves using a register as an intermediate stage in a multi-stage processing pipeline. In simulation, this internal stage can be valuable for observing signal timing and ensuring correct operation; however, it can be invisible to XST. If the register's output is then immediately passed to logic where the register can simply be considered an implementation detail of a higher level function, XST may perceive it as an unnecessary delay element and therefore remove it during optimization. To prevent this, one needs to explicitly make the tool understand that register needs to be present.

Here are several code examples highlighting common scenarios and effective mitigation strategies:

**Example 1: Simple Pipeline Register Removed**

Consider the following Verilog code, designed to introduce a single-cycle delay in a data stream:

```verilog
module delay_pipeline (
  input  wire        clk,
  input  wire        rst,
  input  wire [7:0]  data_in,
  output wire [7:0]  data_out
);

  reg [7:0]  data_stage1;

  always @(posedge clk) begin
    if (rst)
      data_stage1 <= 8'b0;
    else
      data_stage1 <= data_in;
  end

  assign data_out = data_stage1;

endmodule
```

In this simplistic module, `data_stage1` appears to act as a pipeline register. However, XST, upon analysis, will readily recognize that `data_out` is identical to `data_in` delayed by exactly one cycle. Consequently, the register `data_stage1` is optimized away, effectively creating a direct path from input to output. The solution in this case is to ensure there is additional logic or an external interface depending on this register. The simulation would still show the one-cycle delay but the physical implementation would have no register.

**Example 2: Retaining a Pipeline Stage with an Output Port**

To force XST to retain the register, we can add a dedicated output port that is directly tied to this internal stage:

```verilog
module delay_pipeline_debug (
  input  wire        clk,
  input  wire        rst,
  input  wire [7:0]  data_in,
  output wire [7:0]  data_out,
  output wire [7:0]  data_stage1_debug
);

  reg [7:0]  data_stage1;

  always @(posedge clk) begin
    if (rst)
      data_stage1 <= 8'b0;
    else
      data_stage1 <= data_in;
  end

  assign data_out = data_stage1;
  assign data_stage1_debug = data_stage1;

endmodule
```
By adding the `data_stage1_debug` output port, we provide a direct link to an observable output, thereby preventing the tool from removing the internal register. This output port can be used with an integrated logic analyzer (ILA) or similar hardware debugging tool to monitor the register's state. Note that the added output is not directly required by the logical flow of the module; it is only there to prevent optimization of the internal register by making it visible to an output.

**Example 3: Retaining Register using Synthesis Attributes**

Another technique for preserving a register is to use synthesis attributes within your HDL code. These attributes directly guide the behavior of the synthesis tool. Consider this modified module with the keep attribute:

```verilog
module delay_pipeline_attr (
  input  wire        clk,
  input  wire        rst,
  input  wire [7:0]  data_in,
  output wire [7:0]  data_out
);

  reg [7:0]  data_stage1 /* synthesis keep = 1 */;

  always @(posedge clk) begin
    if (rst)
      data_stage1 <= 8'b0;
    else
      data_stage1 <= data_in;
  end

  assign data_out = data_stage1;

endmodule
```

The comment `/* synthesis keep = 1 */` directly instructs the synthesis tool to preserve the register, regardless of the other optimizations. This technique is exceptionally useful for registers that are deemed essential, irrespective of their external connections. It is not necessary to provide a dedicated output; the attribute alone is sufficient to preserve the register. While it is useful, overusing synthesis attributes is discouraged as they can prevent optimizations that increase performance.

In all of these examples, the underlying goal is to influence XST’s understanding of the circuit's purpose. The tool is inherently designed to simplify and optimize, which may not align with our debugging requirements. In the long run, the most effective strategies involve structuring code in ways that explicitly prevent optimization when required, while also ensuring that debug signals are not accidentally optimized away when not in use. This includes using dedicated output ports to observe internal signal states and using synthesis attributes where appropriate, especially for critical registers where removal could lead to functionality issues.

In terms of resource recommendations, I suggest studying the Xilinx Synthesis Guide for your specific Vivado/ISE version. Understand the concept of the 'register balancing' algorithms and the various synthesis attributes that can control how registers are implemented. Another good resource is the Xilinx documentation regarding the use of integrated logic analyzers (ILA) since that often necessitates dedicated output ports for debugging. Exploring Xilinx's example designs can also be useful since they showcase best practices for preventing unintentional register optimization. Furthermore, becoming comfortable with the TCL console of Vivado is incredibly useful, specifically for examining the schematic view of the implementation after synthesis. It allows to visualize exactly what the tool implemented and where registers were added or removed.
