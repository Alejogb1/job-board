---
title: "How to find the maximum value in a Verilog array?"
date: "2025-01-30"
id: "how-to-find-the-maximum-value-in-a"
---
The inherent challenge when finding the maximum value within a Verilog array lies in the absence of built-in high-level functions for such operations, requiring a manual implementation using hardware-describable constructs. Unlike software languages with libraries providing `max` functions, Verilog dictates a structural approach that mirrors the data flow through digital logic. This necessitates creating a comparison-based algorithm directly in hardware description.

The core method involves iteratively comparing elements within the array and retaining the largest value encountered thus far. A register will store the current maximum, initialized to a reasonable starting point, often the first element of the array, or the lowest possible value if dealing with potentially negative quantities. Then, in a controlled, sequential manner, each subsequent element in the array is compared against the current maximum. If the compared element exceeds the current maximum, it replaces the value within the maximum register. This process is repeated until every element in the array has been evaluated, at which point the maximum register will contain the desired maximum value.

This iterative process can be implemented using a for loop within a procedural block, which is commonly an `always` block sensitive to a clock signal. The sequential nature of the comparison operations means that finding the maximum is typically not a single-cycle operation and requires a finite number of clock cycles proportional to the size of the array. Additionally, the register holding the current maximum dictates the width of the data path, which must accommodate the largest potential value present in the array. Care must be taken to manage the indexing of the array, making sure that all array elements are accessed and none accessed beyond the bounds of the array.

The choice of comparison circuitry often relies on basic comparators created from logical operations. A simple comparator outputs one if the first value is larger, zero otherwise. These comparison outputs are then used in conjunction with multiplexers to select either the original maximum or the newly compared value to be written to the maximum register based on the comparator's output. This creates a fundamental hardware building block that iteratively updates the maximum value as the loop progresses.

Here are some code examples with detailed commentary:

**Example 1: Maximum of an 8-element array using blocking assignments:**

```verilog
module max_finder_blocking (
    input clk,
    input [7:0] data_array [7:0], // 8-element array of 8-bit data
    output reg [7:0] max_value,
    output reg valid
);

reg [2:0] index;

always @(posedge clk) begin
    if (index == 0) begin
        max_value <= data_array[0]; //Initialize max to the first element
        valid <= 0; // Reset the valid flag
    end else if (index <= 7) begin
        if (data_array[index] > max_value)
            max_value <= data_array[index];
    end

    if (index < 7) begin
      index <= index + 1;
      valid <= 0;
    end else begin
      index <= 0; // Reset index
      valid <= 1; // Assert valid
    end
end

endmodule
```

*Commentary:* In this example, the `max_finder_blocking` module implements a sequential maximum-finding algorithm using a procedural `always` block with blocking assignments. It utilizes a clock signal for synchronous updates. The `index` register is used to access each element of the `data_array` sequentially. `max_value` stores the current maximum, and `valid` indicates when the maximum of the current array has been computed. Note that the reset is implicit in the behaviour of `index` when `index > 7`, setting the value back to `0`. This is not the ideal design in complex systems, but can be convenient for simple verification or testing scenarios where the array is known to be reset at a higher level.  Blocking assignments are used here because this is a clocked sequential logic block, ensuring the correct logical behaviour across clock edges.

**Example 2: Maximum of an 8-element array using non-blocking assignments:**

```verilog
module max_finder_nonblocking (
    input clk,
    input [7:0] data_array [7:0],
    output reg [7:0] max_value,
    output reg valid
);

reg [2:0] index;
reg [7:0] max_value_reg;

always @(posedge clk) begin
    if (index == 0) begin
        max_value_reg <= data_array[0];
        valid <= 0;
    end else if (index <= 7) begin
        if (data_array[index] > max_value_reg)
            max_value_reg <= data_array[index];
    end

    if (index < 7)
      index <= index + 1;
    else
      index <= 0;

    if (index == 7)
        valid <= 1;
    else
        valid <= 0;
    
    max_value <= max_value_reg;
end

endmodule
```

*Commentary:* This example mirrors the previous one in terms of functionality, however, it demonstrates the use of non-blocking assignments (`<=`). The `max_value` is updated by assigning `max_value_reg` on each clock cycle, rather than within the comparison conditional.  This structure is crucial for robust hardware design, allowing all register updates to occur at the end of the clock cycle without race conditions. The `valid` output is asserted after the last element is checked, also with non-blocking assignments for consistency in clock cycle timing. Using non-blocking assignments avoids inadvertent creation of combinatorial loops, which can result in unpredictable behaviour during simulation and synthesis.

**Example 3: Parameterized Maximum Finding module**

```verilog
module parameterized_max_finder #(
    parameter DATA_WIDTH = 8,
    parameter ARRAY_SIZE = 8
) (
    input clk,
    input [DATA_WIDTH-1:0] data_array [ARRAY_SIZE-1:0],
    output reg [DATA_WIDTH-1:0] max_value,
    output reg valid
);

reg [($clog2(ARRAY_SIZE)-1):0] index;
reg [DATA_WIDTH-1:0] max_value_reg;


always @(posedge clk) begin
    if (index == 0) begin
      max_value_reg <= data_array[0];
      valid <= 0;
    end else if (index < ARRAY_SIZE) begin
        if (data_array[index] > max_value_reg)
            max_value_reg <= data_array[index];
    end

    if (index < (ARRAY_SIZE - 1))
      index <= index + 1;
    else
      index <= 0;

     if (index == (ARRAY_SIZE - 1))
        valid <= 1;
    else
        valid <= 0;
      
     max_value <= max_value_reg;

end
endmodule
```

*Commentary:* This example uses parameters `DATA_WIDTH` and `ARRAY_SIZE` to make the module more flexible. The `index` register is now sized dynamically using `$clog2` to calculate the number of bits required to represent the array size. This enables the module to be instantiated with different array sizes and data widths without having to modify the source code. This approach is far more preferable for reusability in a digital design flow.  This module ensures that no matter how the `ARRAY_SIZE` and `DATA_WIDTH` are set, the maximum finding behaviour is executed correctly.

These examples demonstrate foundational principles for finding the maximum value in a Verilog array, emphasizing the importance of synchronous logic, controlled iterative processes, and the proper use of blocking and non-blocking assignments.

For further study and improvement, I recommend consulting textbooks on digital design using Verilog. These texts will provide comprehensive insight into more advanced techniques, such as pipelined approaches to increasing the throughput of these maximum-finding blocks, as well as design methodologies concerning testbenches and simulation. Additionally, exploring publicly available open-source Verilog implementations will solidify understanding of best practices in hardware description. Finally, research papers that explain the synthesis process that translates Verilog RTL into hardware can aid in understanding how designs are translated into circuits and how different coding styles impact the results in terms of area and performance.
