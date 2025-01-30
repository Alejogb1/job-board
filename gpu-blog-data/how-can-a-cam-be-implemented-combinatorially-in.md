---
title: "How can a CAM be implemented combinatorially in Verilog?"
date: "2025-01-30"
id: "how-can-a-cam-be-implemented-combinatorially-in"
---
Content-Addressable Memories (CAMs), unlike traditional Random Access Memories (RAMs) which access data based on address, operate by searching their entire memory array for a matching data pattern. Implementing this in Verilog combinatorially, meaning the output is directly a function of the current input values without relying on clocked registers, presents design challenges, primarily related to the propagation delays through the matching logic and potential resource consumption. The core issue is ensuring that a match (or non-match) signal is generated with acceptable latency across all memory locations given an input search pattern, without introducing race conditions. My experience developing ASICs for high-throughput packet processing has provided practical insight into addressing these challenges.

The fundamental principle of a combinatorial CAM involves comparing the input search key against every stored word simultaneously. This means that each memory location requires comparison logic, and the outputs of these comparisons must be combined to determine the overall match result. Specifically, this is not a sequential search but rather a fully parallel evaluation. If an exact match is found, it will trigger the corresponding output. The implementation can be broken down into several key steps: Memory Storage, Comparison Logic, Priority Encoding (if needed), and Output Generation.

For simplicity, we'll start with a basic CAM without priority encoding; i.e., if multiple matches exist, any one match or all of them could be indicated. The first component to implement is the storage itself. In a combinational implementation, the data must be directly accessible to the comparator. This will typically necessitate using Verilog assign statements and a large, sparse array rather than registers which would introduce a delay.

```verilog
module basic_cam #(
  parameter DATA_WIDTH = 8,
  parameter MEMORY_DEPTH = 4
)(
  input [DATA_WIDTH-1:0] search_key,
  output reg [MEMORY_DEPTH-1:0] match_output
);

  // Initialize the memory to some test values.
  // In a real implementation, this would likely come from
  // some external interface or an initialization process.
  localparam [DATA_WIDTH-1:0] mem[MEMORY_DEPTH-1:0] =
    '{8'h55, 8'hAA, 8'h3C, 8'h1D};

  integer i;

  always @(*) begin
    match_output = 0; // Assume no match initially
    for (i=0; i<MEMORY_DEPTH; i=i+1) begin
        if (mem[i] == search_key) begin
            match_output[i] = 1'b1;
        end
    end
  end
endmodule
```
In the preceding example, the `mem` localparam statically defines the stored values. The `always @(*)` block continuously monitors any change to the `search_key` input. It iterates through each memory location and, if a match is found, sets the corresponding bit in the `match_output`. This provides a straightforward, if not optimized, combinatorial CAM behavior. Note that using a loop inside of always@(*) can result in inefficient logic synthesis. The purpose here is to illustrate the overall function. A better way would be to unroll the loop during synthesis using generates.

A more efficient approach is to use generate statements in Verilog in lieu of for loops within a combinational context. This unrolls the logic at compile time. This will yield a much more efficient and predictable implementation.
```verilog
module efficient_cam #(
  parameter DATA_WIDTH = 8,
  parameter MEMORY_DEPTH = 4
)(
  input [DATA_WIDTH-1:0] search_key,
  output [MEMORY_DEPTH-1:0] match_output
);

  localparam [DATA_WIDTH-1:0] mem[MEMORY_DEPTH-1:0] =
  '{8'h55, 8'hAA, 8'h3C, 8'h1D};
  genvar i;

  generate
    for (i=0; i<MEMORY_DEPTH; i=i+1) begin: match_logic
      assign match_output[i] = (mem[i] == search_key);
    end
  endgenerate
endmodule
```
This revised example uses a `generate` block and `assign` statements, achieving the same functionality as the first example, but without the resource overhead of a loop operating dynamically within a combinatorial always block. Each `match_output[i]` bit directly reflects the comparison outcome for memory location `i`. This approach scales with memory size, although extremely large memories might need segmentation to respect synthesis tool resource limitations.

For many applications, a simple "any match" indication is insufficient. One typically requires a priority encoder to indicate the address of the highest-priority matching location. This requires introducing additional combinatorial logic to perform prioritization.

```verilog
module prioritized_cam #(
  parameter DATA_WIDTH = 8,
  parameter MEMORY_DEPTH = 4
)(
  input [DATA_WIDTH-1:0] search_key,
  output reg [MEMORY_DEPTH-1:0] match_output,
  output reg [$clog2(MEMORY_DEPTH)-1:0] match_address,
  output reg match_found
);

  localparam [DATA_WIDTH-1:0] mem[MEMORY_DEPTH-1:0] =
  '{8'h55, 8'hAA, 8'h3C, 8'h1D};
  genvar i;
  wire [MEMORY_DEPTH-1:0] internal_matches;

  generate
    for (i=0; i<MEMORY_DEPTH; i=i+1) begin: match_logic
      assign internal_matches[i] = (mem[i] == search_key);
    end
  endgenerate

  always @(*) begin
    match_output = internal_matches;
    match_address = 0;
    match_found = 0;

    for (i=MEMORY_DEPTH-1; i>=0; i=i-1) begin
      if (internal_matches[i]) begin
        match_address = i;
        match_found = 1;
        break;
      end
    end
  end
endmodule
```
Here, the `internal_matches` wire combines the individual match results as in the previous example. However, within the `always @(*)` block, we prioritize matches. We iterate through the `internal_matches` wire in reverse order. The first match found (corresponding to the highest index) is used to set `match_address`. The `match_found` flag indicates whether any match was detected at all. Note that the priority is defined by the order of the search (highest indexed word takes precedence). If no matches are found, the outputs will default to zero.

Implementing CAMs combinatorially involves a fundamental trade-off between latency and resource consumption. In the examples provided above, increasing the memory depth directly results in an increase in the required resources as well as an increase in propagation delay, as the comparison and priority encoding logic grows. A balance must be found to achieve required functionality within the limits of the target hardware.

To achieve more optimal results, the following are recommended for additional study:

*   **Synthesis Reports Analysis**: Familiarize yourself with the synthesis reports generated by your tools. These provide crucial information on resource usage (LUTs, FFs, etc.) and timing analysis (propagation delays, setup/hold times). These reports reveal how the Verilog code translates into hardware and allow you to pinpoint performance bottlenecks.

*   **Logic Optimization Techniques:** The core of optimizing any CAM implementation involves streamlining the combinational logic. Explore techniques such as Karnaugh maps and Quine-McCluskey minimization to reduce logic gate usage when creating comparison or priority logic. This can be particularly useful with more complex comparators.

*   **Architecture Trade-offs:** When implementing very large CAMs, consider architectural trade-offs such as segmentation and pipelining. Segmenting the CAM into smaller blocks and introducing pipelined stages can reduce latency and improve scalability.

*   **Understanding Target Hardware**: Recognize the underlying limitations of your target hardware, such as FPGA LUT sizes or ASIC cell libraries. Adjust your coding and design approaches to maximize resource utilization and performance within these limitations.

In summary, constructing a combinatorial CAM in Verilog requires understanding the parallel nature of CAM operation and carefully implementing the memory, comparison, and priority logic to maintain acceptable performance and resource utilization. The provided code demonstrates the core concepts of this implementation, and further work is needed to optimize a specific use case.
