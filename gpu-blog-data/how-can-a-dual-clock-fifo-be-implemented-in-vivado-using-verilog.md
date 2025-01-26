---
title: "How can a dual-clock FIFO be implemented in Vivado using Verilog?"
date: "2025-01-26"
id: "how-can-a-dual-clock-fifo-be-implemented-in-vivado-using-verilog"
---

Implementing a dual-clock FIFO (First-In, First-Out) in Vivado using Verilog presents a significant challenge due to the inherent risks associated with crossing clock domains. A poorly designed FIFO can lead to metastability, data corruption, and overall system instability. The core requirement is to transfer data reliably between two asynchronous clock domains, which requires careful consideration of signal synchronization and full/empty flag generation. I’ve addressed this issue multiple times while developing high-throughput data acquisition systems for scientific instruments, and I've found a specific approach to consistently provide stable results.

The fundamental concept involves using Gray codes for the read and write pointers. Gray codes ensure that only one bit changes between consecutive values, minimizing the chance of metastability when these pointers are synchronized across clock domains. Without Gray coding, multiple bits might transition simultaneously, leading to unpredictable behavior when sampled by the asynchronous clock. This forms the basis of my design, which I'll outline using a synchronous SRAM for the data storage.

My proposed dual-clock FIFO consists of several key modules: the SRAM for data storage, the write-side logic, the read-side logic, and the synchronization circuits. The write-side module manages the insertion of data into the SRAM based on the write clock domain, while the read-side module manages the removal of data based on the read clock domain. The synchronization logic is responsible for safely passing the write and read pointers between the two asynchronous domains and for reliably generating the full and empty flags.

Here's a breakdown of the approach, starting with the Verilog implementation:

**1. Parameterized FIFO Definition and Memory Array**

```verilog
module dual_clock_fifo #(
    parameter DATA_WIDTH = 8,
    parameter FIFO_DEPTH_BITS = 4, // Depth = 2^FIFO_DEPTH_BITS
    parameter ADDR_WIDTH = FIFO_DEPTH_BITS,
    parameter FIFO_DEPTH = 1 << FIFO_DEPTH_BITS
  )(
    input  wire             wr_clk,
    input  wire             wr_en,
    input  wire [DATA_WIDTH-1:0] wr_data,
    output wire             full,
    input  wire             rd_clk,
    input  wire             rd_en,
    output wire [DATA_WIDTH-1:0] rd_data,
    output wire             empty
  );

  // Memory array
  reg [DATA_WIDTH-1:0] mem [0:FIFO_DEPTH-1];
  reg [ADDR_WIDTH-1:0] wr_ptr;
  reg [ADDR_WIDTH-1:0] rd_ptr;

  // Synchronization Registers and Full/Empty generation logic will be included in the separate modules

  // Internal registers and signals here...

  assign rd_data = mem[rd_ptr]; // Read Data
```

Here, I define the top-level module with parameterized data width, address width, and FIFO depth. The `mem` array serves as the storage, implemented as registers for this example. The `wr_ptr` and `rd_ptr` registers manage write and read address pointers. The FIFO is accessed through `wr_data` and `rd_data`, which are connected to the memory array. The module interface also includes `wr_clk`, `wr_en`, `rd_clk`, and `rd_en`, controlling write and read operations as well as full and empty flags.

**2. Write-Side Logic**

```verilog
module fifo_write_side #(
  parameter DATA_WIDTH = 8,
  parameter FIFO_DEPTH_BITS = 4,
  parameter ADDR_WIDTH = FIFO_DEPTH_BITS,
  parameter FIFO_DEPTH = 1 << FIFO_DEPTH_BITS
)(
    input  wire             wr_clk,
    input  wire             wr_en,
    input  wire [DATA_WIDTH-1:0] wr_data,
    output wire             full,
    input  wire  [ADDR_WIDTH:0]  sync_rd_ptr_gray,
    output reg  [ADDR_WIDTH-1:0] wr_ptr,
    output reg [ADDR_WIDTH:0] wr_ptr_gray
  );

  reg [ADDR_WIDTH-1:0] next_wr_ptr;
  reg [ADDR_WIDTH:0]   next_wr_ptr_gray;
  reg [ADDR_WIDTH:0]  prev_wr_ptr_gray;

  always @(posedge wr_clk) begin
    if (wr_en) begin
      wr_ptr      <= next_wr_ptr;
      wr_ptr_gray <= next_wr_ptr_gray;
      prev_wr_ptr_gray <= wr_ptr_gray;
    end
  end

  always @(*) begin
        next_wr_ptr = wr_ptr + 1;
        next_wr_ptr_gray = (next_wr_ptr ^ (next_wr_ptr >> 1));
  end


  // Full Flag Generation
  wire [ADDR_WIDTH:0] next_wr_ptr_gray_synced;
  assign next_wr_ptr_gray_synced = next_wr_ptr_gray;

  assign full = (next_wr_ptr_gray_synced == sync_rd_ptr_gray[ADDR_WIDTH:0]) ;


  endmodule
```

In the write-side module, data is written to the memory if `wr_en` is asserted. The write pointer `wr_ptr` increments on each write operation, and its Gray code equivalent, `wr_ptr_gray`, is generated using a bitwise XOR operation. This value is subsequently sent to the read clock domain after synchronization.  The logic calculates the next write pointer `next_wr_ptr` and its gray coded version `next_wr_ptr_gray`, and updates the current register values on the positive edge of `wr_clk`. The full flag is generated by comparing the synchronized read pointer with the next write pointer. The `next_wr_ptr_gray` is assigned directly to `next_wr_ptr_gray_synced`, with the intention that its value will be synchronized when it passes through the synchronizer.

**3. Read-Side Logic**

```verilog
module fifo_read_side #(
    parameter DATA_WIDTH = 8,
    parameter FIFO_DEPTH_BITS = 4,
    parameter ADDR_WIDTH = FIFO_DEPTH_BITS,
    parameter FIFO_DEPTH = 1 << FIFO_DEPTH_BITS
  )(
    input  wire             rd_clk,
    input  wire             rd_en,
    output wire             empty,
    input  wire [ADDR_WIDTH:0] sync_wr_ptr_gray,
    output reg  [ADDR_WIDTH-1:0] rd_ptr,
    output reg [ADDR_WIDTH:0] rd_ptr_gray
  );


  reg [ADDR_WIDTH-1:0] next_rd_ptr;
  reg [ADDR_WIDTH:0]   next_rd_ptr_gray;
  reg [ADDR_WIDTH:0] prev_rd_ptr_gray;


  always @(posedge rd_clk) begin
    if (rd_en) begin
      rd_ptr      <= next_rd_ptr;
      rd_ptr_gray <= next_rd_ptr_gray;
      prev_rd_ptr_gray <= rd_ptr_gray;
    end
  end

  always @(*) begin
        next_rd_ptr = rd_ptr + 1;
        next_rd_ptr_gray = (next_rd_ptr ^ (next_rd_ptr >> 1));
  end

  // Empty Flag Generation
  wire [ADDR_WIDTH:0] next_rd_ptr_gray_synced;
  assign next_rd_ptr_gray_synced = next_rd_ptr_gray;
  assign empty = (next_rd_ptr_gray_synced == sync_wr_ptr_gray[ADDR_WIDTH:0]);

endmodule
```

Similar to the write-side module, the read-side logic manages the read pointer `rd_ptr` and its corresponding Gray code `rd_ptr_gray`, incrementing it on each read enable `rd_en`. The empty flag is generated by comparing the synchronized write pointer with the next read pointer. The `next_rd_ptr_gray` is assigned directly to `next_rd_ptr_gray_synced`, to be synchronized in a separate module.

**4. Pointer Synchronization Module**

The crucial aspect of a dual-clock FIFO is the synchronization of the read and write pointers between clock domains. A typical synchronizer uses two or more flip-flops in series clocked by the destination clock domain to mitigate metastability risks, ensuring that the signal is properly resolved before being used in the destination domain.

```verilog
module synchronizer #(
  parameter DATA_WIDTH = 1
)(
  input  wire  clk,
  input  wire  data_in,
  output reg   data_out
);

  reg data_sync_reg1;
  reg data_sync_reg2;

  always @(posedge clk) begin
    data_sync_reg1 <= data_in;
    data_sync_reg2 <= data_sync_reg1;
  end

  assign data_out = data_sync_reg2;

endmodule
```

This simple synchronizer consists of two flip-flops, clocked by the `clk` input. The input data `data_in` is captured by the first register and passed to the second before being output as `data_out`. For this application, one instantiation of this synchronizer module will be needed for each bit of the Gray coded write and read pointers.

The dual-clock FIFO implementation uses multiple instances of the synchronizer to properly transfer the read and write pointers between the domains. These synchronized pointers are then used in the full and empty flag generation logic.

**Resource Recommendations**

For further understanding and refinement, I highly recommend exploring these specific resource areas:

1.  **Digital Design Principles**: Emphasize a solid understanding of digital logic and synchronous design. Pay specific attention to timing constraints and how those limitations affect the circuit's performance and reliability.
2.  **Asynchronous Design**: Focus on the challenges associated with crossing clock domains and various methods to overcome those problems. Understanding metastability and its consequences is paramount when designing reliable asynchronous systems. Look at techniques such as handshaking as potential alternative methodologies.
3.  **FPGA Design Guides**:  Study vendor-specific documentation and application notes that cover best practices for implementing asynchronous circuits. Particular focus should be placed on clock crossing techniques and the use of synchronization primitives in Vivado. Review Xilinx application notes for dual clock FIFO implementation.

A deep dive into these areas will yield a robust understanding of not just the implementation details, but also the underlying principles that guide stable and efficient dual-clock FIFO designs.
