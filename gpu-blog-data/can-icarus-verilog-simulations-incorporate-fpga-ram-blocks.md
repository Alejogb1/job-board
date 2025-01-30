---
title: "Can Icarus Verilog simulations incorporate FPGA RAM blocks?"
date: "2025-01-30"
id: "can-icarus-verilog-simulations-incorporate-fpga-ram-blocks"
---
I have encountered situations where simulating FPGA RAM blocks within an Icarus Verilog environment presented challenges, particularly concerning accuracy and resource representation. The core issue lies in Icarus Verilog's nature as a simulator focused on behavioral and register-transfer level (RTL) abstractions, not low-level hardware primitives like FPGA-specific RAM. Directly incorporating a vendor's RAM primitive as a module, for example, will typically lead to compilation errors or simulations that do not reflect the actual hardware behavior.

Icarus Verilog primarily interprets Verilog code that describes the logical function of a circuit. It is excellent at verifying the logical flow of data, control paths, and basic arithmetic operations within a design. However, FPGA RAM blocks are intricate, often exhibiting behaviors dependent on the specific silicon architecture, timing constraints, and configuration parameters, which standard Verilog cannot fully encapsulate. A straightforward instantiation of a vendor-provided RAM module will, at best, act as an equivalent memory model, not as the actual silicon component. Therefore, a more nuanced approach is needed.

To properly handle RAM simulations within Icarus Verilog, one typically needs to abstract away the hardware-specific nuances. This usually involves using a behavioral model that replicates the functional characteristics of the RAM. The model needs to capture the critical aspects, including read and write operations, addressability, data width, and sometimes, synchronous and asynchronous characteristics. It is crucial to understand that while functionally accurate, this model deviates from the physical RAM implementation. It cannot accurately simulate delays intrinsic to the specific RAM cell or its interaction with the FPGA routing fabric.

Essentially, when simulating, I do not simulate the ‘real’ RAM. Rather, I simulate a Verilog module behaving *as if* it were the RAM. This simulation is beneficial for validating the overall algorithm, verifying data integrity, and understanding how the rest of the design interfaces with the memory, but should not be regarded as a timing-accurate representation of the FPGA hardware.

The abstraction I usually implement uses a simple behavioral RAM, implemented in Verilog, that captures the core functionality of an FPGA's block RAM. This module does not necessarily emulate every configurable feature, but typically includes address decoding, data storage using Verilog memory arrays, read and write enable inputs, and a clock signal. This approach has consistently provided acceptable results when performing functional verification.

Here are a few code examples to illustrate the concept:

**Example 1: Basic Synchronous RAM Model**

This example demonstrates a simple synchronous RAM with single-port access. This is the basic style I use for most initial simulations, especially when timing isn’t my primary focus.

```verilog
module sync_ram (
    input clk,
    input we,
    input [7:0] addr,
    input [15:0] data_in,
    output reg [15:0] data_out
);

  reg [15:0] mem [0:255]; // 256 locations of 16-bit width

  always @(posedge clk) begin
    if (we) begin
      mem[addr] <= data_in;
    end
    data_out <= mem[addr];
  end

endmodule
```
**Commentary:** This module implements a synchronous RAM with a 256 location memory array. The `we` input signal enables writing to memory. The `addr` input determines the memory location. The `data_in` input provides the data to write. The `data_out` register always holds the value at the address location, and this value is synchronously updated with the rising edge of the clock. This simplified model omits features such as output register delays, and read-before-write behavior, but it accurately captures the core read and write functionality within a simulation environment.

**Example 2: Dual-Port RAM Model**

When my design requires simultaneous reads and writes, I implement a dual-port RAM. Here, two different address buses allow concurrent operations.

```verilog
module dual_port_ram (
  input clk,
  input we_a,
  input [7:0] addr_a,
  input [15:0] data_in_a,
  output reg [15:0] data_out_a,

  input we_b,
  input [7:0] addr_b,
  input [15:0] data_in_b,
  output reg [15:0] data_out_b
  );

  reg [15:0] mem [0:255]; // 256 locations of 16-bit width

  always @(posedge clk) begin
    if (we_a) begin
        mem[addr_a] <= data_in_a;
    end
    data_out_a <= mem[addr_a];

     if (we_b) begin
        mem[addr_b] <= data_in_b;
    end
    data_out_b <= mem[addr_b];

  end

endmodule
```

**Commentary:**  This model showcases the implementation of a dual-port memory. Port A uses the `addr_a`, `data_in_a`, `we_a` and `data_out_a` signals, and port B uses the `addr_b`, `data_in_b`, `we_b` and `data_out_b` signals. This configuration allows for simultaneous read and write operations, or two simultaneous read operations. This model accurately represents the functional behavior of a dual-port RAM, allowing me to verify that data management with multiple read/write interfaces are functioning as intended.

**Example 3:  RAM Model with Initialization**

Sometimes, I use a RAM model that can be initialized with a starting set of data, which helps with the preloading data into specific memory locations for simulation scenarios.

```verilog
module init_ram (
  input clk,
  input we,
  input [7:0] addr,
  input [15:0] data_in,
  output reg [15:0] data_out,
  input [15:0] init_data [0:255] // Initial data

  );

  reg [15:0] mem [0:255]; // 256 locations of 16-bit width
  integer i;


  initial begin
    for(i=0; i < 256; i=i+1) begin
        mem[i] = init_data[i];
    end
  end

  always @(posedge clk) begin
    if (we) begin
      mem[addr] <= data_in;
    end
    data_out <= mem[addr];
  end

endmodule
```

**Commentary:** The key difference here is the inclusion of an `init_data` input. The `initial` block loads this initial data into the `mem` register during simulation initialization. This capability is incredibly useful for test benches where pre-defined data should exist in the memory before the simulation begins, and allows to perform checks based on expected initial memory content. The rest of the model performs the same read/write operations as the earlier examples.

When considering external resources for learning more about modeling RAM in Verilog, I find it beneficial to review standard Verilog language reference manuals, focusing on memory array syntax and timing control. Books focusing on digital design often provide background on memory architecture and its modeling considerations. Online Verilog tutorials frequently include detailed examples of memory modeling, particularly those centered around hardware description languages and simulation. Furthermore, examining the documentation for different FPGA development tools can reveal their methodologies for RAM instantiation during synthesis. Lastly, some academic articles specifically address the topic of high-level modeling of hardware components, which may provide deeper theoretical insights.
