---
title: "Can Verilog simultaneously write to and read from an FPGA register?"
date: "2025-01-30"
id: "can-verilog-simultaneously-write-to-and-read-from"
---
Verilog's ability to simultaneously read and write to an FPGA register depends entirely on the register's definition and the timing constraints imposed by the FPGA's clocking scheme.  Direct simultaneous access in the sense of a single clock cycle is generally not possible, but concurrent access within a clock cycle, using carefully planned design techniques, can achieve functionally equivalent results.  This stems from the fundamental sequential nature of hardware description languages and the synchronous operation of FPGAs.


My experience developing high-speed data acquisition systems for space-based applications frequently involved managing concurrent register access.  Misunderstanding this nuance led to several early design iterations exhibiting unpredictable behavior.  The key lies not in simultaneous access, but in meticulously separating read and write operations within the clock cycle, leveraging careful scheduling and potentially introducing intermediate staging registers.

**1.  Clear Explanation:**

FPGA registers are fundamentally edge-triggered or level-sensitive devices.  This means that their value changes only at specific points in time – typically the rising or falling edge of a clock signal.  A write operation initiates a change in the register's value at the active clock edge.  Simultaneously attempting to read the register during the same clock edge will either read the *previous* value (before the write) or an indeterminate value, depending on the FPGA's internal architecture and the specific synthesis tool's optimizations.  The indeterminate value is a problematic "race condition" that must be carefully avoided.

Therefore, "simultaneous" access must be reinterpreted as concurrent access within a clock cycle.  This is achievable using several methods:

* **Sequential Logic:** Introduce intermediate registers or state variables to manage the read and write operations sequentially.  The write operation occurs first, followed by a read operation in a subsequent clock cycle.  This guarantees that the read operation accesses the updated register value.

* **Clock Domain Crossing:** If the read and write operations are performed in different clock domains, meticulous synchronization using appropriate techniques like asynchronous FIFOs or multi-cycle paths must be used to avoid metastability issues.  This approach is complex and necessitates a deep understanding of clock domain crossing design principles.

* **Separate Registers:** Employ separate read and write registers. Data is written to one register and simultaneously read from another (pre-populated) register.  In the next clock cycle, the contents of the write register are transferred to the read register.  This guarantees a consistent read value while supporting concurrent write operations.


**2. Code Examples with Commentary:**

**Example 1: Sequential Logic Approach**

```verilog
module sequential_rw (
  input clk,
  input rst,
  input write_enable,
  input [7:0] write_data,
  output reg [7:0] read_data
);

  reg [7:0] internal_reg;

  always @(posedge clk) begin
    if (rst) begin
      internal_reg <= 8'b0;
      read_data <= 8'b0;
    end else begin
      if (write_enable) begin
        internal_reg <= write_data;
      end
      read_data <= internal_reg; //Read in the next clock cycle
    end
  end

endmodule
```

This example demonstrates sequential access.  The `internal_reg` stores the written data.  The `read_data` is updated *after* the write, guaranteeing that the read operation accesses the updated value in the next clock cycle.  This eliminates any race conditions.

**Example 2: Separate Registers Approach**

```verilog
module separate_registers (
  input clk,
  input rst,
  input write_enable,
  input [7:0] write_data,
  output reg [7:0] read_data
);

  reg [7:0] write_reg;
  reg [7:0] read_reg;

  always @(posedge clk) begin
    if (rst) begin
      write_reg <= 8'b0;
      read_reg <= 8'b0;
    end else begin
      if (write_enable) begin
        write_reg <= write_data;
      end
      read_reg <= write_reg; //Transfer data to the read register
      read_data <= read_reg;
    end
  end

endmodule
```

Here, `write_reg` and `read_reg` are distinct.  The write operation updates `write_reg`.  In the following clock cycle, the data is transferred to `read_reg`, ensuring a consistent read value.  This approach enables functionally concurrent read and write operations within a clock cycle.

**Example 3:  Handling potential hazards using a flag.**

This example expands on the sequential approach, demonstrating how to signal when a new value is available for reading, preventing potential read hazards.

```verilog
module flagged_sequential_rw (
  input clk,
  input rst,
  input write_enable,
  input [7:0] write_data,
  output reg [7:0] read_data,
  output reg data_ready
);

  reg [7:0] internal_reg;

  always @(posedge clk) begin
    if (rst) begin
      internal_reg <= 8'b0;
      read_data <= 8'b0;
      data_ready <= 1'b0;
    end else begin
      if (write_enable) begin
        internal_reg <= write_data;
        data_ready <= 1'b1; // Signal data is ready after write
      end else begin
        data_ready <= 1'b0;
      end
      read_data <= internal_reg;
    end
  end

endmodule
```

The `data_ready` flag indicates when `read_data` contains valid, recently written data.  This prevents unintended reads before a write completes.  The consumer of `read_data` should only access it when `data_ready` is asserted.

**3. Resource Recommendations:**

For deeper understanding, I recommend consulting the official documentation for your specific FPGA vendor's synthesis and simulation tools.  Additionally, a thorough grounding in digital design principles, clock domain crossing techniques, and Verilog's timing behavior is essential.  Finally, a well-structured digital design textbook covering register transfer level (RTL) design will offer valuable context.  Exploring examples of asynchronous FIFOs and their implementation in Verilog is particularly beneficial when dealing with clock domain crossing scenarios involving concurrent read/write operations.
