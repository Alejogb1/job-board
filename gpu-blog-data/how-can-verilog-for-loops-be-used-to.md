---
title: "How can Verilog for loops be used to read data from a ROM block?"
date: "2025-01-30"
id: "how-can-verilog-for-loops-be-used-to"
---
Within digital design, memory access, particularly from read-only memory (ROM), often necessitates structured iteration to sequentially retrieve data bytes or words. Verilog's `for` loop provides a mechanism to implement such iterative reads, although its synthesis behavior requires careful consideration. The primary challenge lies in understanding that, unlike software programming, Verilog `for` loops are typically unrolled by synthesis tools, creating replicated hardware structures rather than dynamic loop execution at runtime. This impacts how we address memory locations within the ROM and the resulting hardware footprint.

Fundamentally, a Verilog `for` loop employed for ROM access defines a range of indices that translate into address inputs to the ROM. Each loop iteration typically fetches a single data word from the addressed location, which can then be assigned to a register or otherwise processed. Given that the loop is unrolled at synthesis time, its use is most appropriate when the number of iterations is a compile-time constant, and the address calculations can be statically determined. This contrasts with dynamically indexed arrays in software, where the loop's execution count can be a runtime variable.

To illustrate the process, consider a scenario where I designed a microcontroller peripheral responsible for displaying character data stored in ROM on an LCD. The ROM was defined to have an address space large enough to store an entire font set. I used a Verilog module to interface with this ROM and retrieve a string of character data. The design aimed to perform data fetch by iterating through a known set of address ranges, thus leveraging the deterministic nature of loop unrolling.

The following code snippets detail specific implementations showcasing different strategies in ROM access with `for` loops.

**Code Example 1: Reading a fixed length string**

This example demonstrates reading a fixed length string from the ROM. The loop's bounds are static, meaning that the loop will iterate over a pre-defined, compile-time known number of cycles.

```verilog
module rom_read_fixed_string (
  input  wire         clk,
  input  wire         reset,
  output reg [7:0]    char_data,  // Output character data
  output reg          data_valid   // Flag indicating data is valid
);

  parameter ROM_ADDR_WIDTH = 8;
  parameter ROM_DATA_WIDTH = 8;
  parameter START_ADDR = 8'h20; // Start address of the string in the ROM
  parameter STRING_LEN = 10;   // Length of the string to read

  reg [ROM_ADDR_WIDTH-1:0] rom_addr;
  reg [ROM_DATA_WIDTH-1:0] rom_data;

  // Instantiate the ROM memory block (Assume a module named 'rom_block')
  rom_block #(
    .ADDR_WIDTH(ROM_ADDR_WIDTH),
    .DATA_WIDTH(ROM_DATA_WIDTH)
  ) u_rom (
    .addr   (rom_addr),
    .data   (rom_data)
  );

  reg [3:0] char_index;
  reg       busy;

  always @(posedge clk or posedge reset) begin
      if(reset) begin
          char_index <= 4'b0;
          char_data <= 8'b0;
          data_valid <= 1'b0;
          busy <= 1'b0;
      end else begin
          if(!busy) begin
            busy <= 1'b1;
            for (char_index = 4'b0; char_index < STRING_LEN; char_index = char_index + 1) begin
               rom_addr <= START_ADDR + char_index;  // Address calculation for current char.
               char_data <= rom_data;
               data_valid <= 1'b1;
            end
            busy <= 1'b0;
          end
          else begin
            data_valid <= 1'b0;
          end
      end
    end

endmodule

```

**Commentary on Code Example 1:**

This module illustrates a simple approach to reading a string of characters from a ROM with a `for` loop.  The loop iterates from 0 up to a compile-time defined constant, `STRING_LEN`.  The ROM address is calculated by adding the current loop index to the `START_ADDR`. Each iteration of the unrolled loop, after synthesis, represents a distinct read cycle of the ROM. The `busy` flag prevents continuous execution of the loop. Although simple, this example demonstrates a key aspect of Verilog loops for memory access:  the loop executes in zero-time in simulation since it models a hardware construct where the address lines are calculated in parallel. In real hardware, the execution of this loop is dependent on the clock cycles. This is a critical distinction for anyone coming from software-oriented languages. It’s also worth noting that for each iteration, the `rom_addr` value changes, and `rom_data` takes the value after the next read cycle from the instantiated `rom_block`. This method works well when the number of read operations is low, but if the `STRING_LEN` is very large, this will result in a lot of duplicated hardware at synthesis. The data and validity signals are asserted only in each clock cycle.

**Code Example 2: Reading data into a register array**

This snippet reads data from ROM and stores it within an internal register array, thereby demonstrating a scenario where the data is kept within the device for further manipulation.

```verilog
module rom_read_array (
  input  wire         clk,
  input  wire         reset,
  output reg [7:0]    data_array [0:9], // Output array of data
  output reg          read_done // Flag indicating all reads are completed.
);

  parameter ROM_ADDR_WIDTH = 8;
  parameter ROM_DATA_WIDTH = 8;
  parameter START_ADDR = 8'h40;
  parameter ARRAY_SIZE = 10;

  reg [ROM_ADDR_WIDTH-1:0] rom_addr;
  reg [ROM_DATA_WIDTH-1:0] rom_data;

  // Instantiate the ROM block
  rom_block #(
    .ADDR_WIDTH(ROM_ADDR_WIDTH),
    .DATA_WIDTH(ROM_DATA_WIDTH)
  ) u_rom (
    .addr   (rom_addr),
    .data   (rom_data)
  );

  reg [3:0] index;
  reg busy;

  always @(posedge clk or posedge reset) begin
      if(reset) begin
          read_done <= 1'b0;
          busy <= 1'b0;
      end else begin
           if(!busy) begin
                busy <= 1'b1;
                for (index = 4'b0; index < ARRAY_SIZE; index = index + 1) begin
                    rom_addr <= START_ADDR + index;
                    data_array[index] <= rom_data;
                end
                read_done <= 1'b1;
                busy <= 1'b0;
            end else begin
              read_done <= 1'b0;
           end
        end
  end
endmodule
```

**Commentary on Code Example 2:**

Here, the `for` loop iterates over `ARRAY_SIZE` and reads data from the ROM into the register array `data_array`. Similar to the first example, the loop is unrolled, and each iteration corresponds to hardware that performs an address calculation and read operation. The register `data_array` stores values read from the `rom_data`. The `read_done` signal indicates the completion of all the read operations. The loop here does not depend on any other asynchronous value; instead, the `busy` flag makes sure the loop is not executed in a continuous loop. It is worth noting that while the `data_array` values are updated in simulation using a zero-time execution, during hardware implementation, the reads are still a sequential operation that will occur one clock cycle at a time. The read operation is still limited by the memory access speed and the system’s clock frequency.

**Code Example 3: Conditional ROM Reads within a loop**

This last example introduces a conditional read within the loop, where the ROM read only happens if a condition is met, while also illustrating how variables defined outside of a for loop can be used inside of it to manipulate the memory address.

```verilog
module rom_read_conditional (
    input wire clk,
    input wire reset,
    input wire read_enable,
    output reg [7:0] out_data
);

  parameter ROM_ADDR_WIDTH = 8;
  parameter ROM_DATA_WIDTH = 8;
  parameter START_ADDR  = 8'h60;
  parameter NUM_READS   = 5;

  reg [ROM_ADDR_WIDTH-1:0] rom_addr;
  reg [ROM_DATA_WIDTH-1:0] rom_data;

  // Instantiate the ROM block
  rom_block #(
    .ADDR_WIDTH(ROM_ADDR_WIDTH),
    .DATA_WIDTH(ROM_DATA_WIDTH)
  ) u_rom (
    .addr   (rom_addr),
    .data   (rom_data)
  );

  reg [3:0] i;
  reg [7:0] temp_data;
  reg busy;
  reg [7:0] current_addr;

  always @(posedge clk or posedge reset) begin
      if (reset) begin
          busy <= 1'b0;
          out_data <= 8'b0;
          current_addr <= START_ADDR;
      end else begin
        if (read_enable && !busy) begin
          busy <= 1'b1;
          for(i = 0; i < NUM_READS; i = i+1) begin
             rom_addr <= current_addr;
             if (rom_data > 8'h20) begin
                out_data <= rom_data;
              end
              current_addr <= current_addr + 1;
           end
          busy <= 1'b0;
        end
    end
  end

endmodule
```

**Commentary on Code Example 3:**

This module introduces conditional ROM reads within the `for` loop. The address offset from the base is incremented within the loop. The `current_addr` variable is defined outside of the loop and updated within. The loop iterates `NUM_READS` times, reading data from the ROM at the `current_addr`. Data is only output if its value is greater than 0x20. The conditional statement is implemented in hardware with combinational logic, while the loop is unrolled and takes multiple clock cycles in real hardware. This example demonstrates a more sophisticated usage of loops, with the address to the ROM being dynamically altered inside the for loop.

**Resource Recommendations**

For a deeper understanding of Verilog and its application in hardware design, I would recommend exploring the following resources:

1.  **Textbooks on Digital Design:** Look for textbooks that cover both the theoretical foundations of digital logic and the practical aspects of using hardware description languages (HDLs) such as Verilog. Pay particular attention to sections discussing synthesis, memory architectures, and state machine design.

2.  **Online Courses:** Many online education platforms offer comprehensive courses on digital design using Verilog. Search for courses that explicitly address the synthesis process and its impact on hardware implementation. These courses often include practical projects that can solidify your understanding.

3. **Vendor Documentation:** Semiconductor vendors provide extensive documentation for their synthesis tools. Consult the documentation for your specific toolchain to understand how it handles `for` loops and other Verilog constructs. Focus on resources related to timing analysis, resource utilization, and synthesis optimization.

In summary, employing Verilog `for` loops for ROM data access is a valid approach, provided one understands the implications of loop unrolling during synthesis. Proper loop bounds and address calculation are critical, while careful consideration of asynchronous behavior and clock frequency are necessary for real hardware implementations. The three examples and resource suggestions presented here offer a foundation for employing such techniques effectively.
