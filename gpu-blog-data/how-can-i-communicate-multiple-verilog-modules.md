---
title: "How can I communicate multiple Verilog modules?"
date: "2025-01-30"
id: "how-can-i-communicate-multiple-verilog-modules"
---
The fundamental challenge in designing complex digital systems with Verilog lies not just in creating individual modules, but in establishing robust and reliable communication pathways between them. This inter-module communication is crucial for realizing the desired functionality. It can be achieved through several methods, each with its own set of advantages and disadvantages. Based on my experience developing custom cryptographic accelerators and high-speed networking interfaces, I find careful consideration of the communication strategy significantly impacts design performance, resource utilization, and ultimately, verification complexity.

The most common and generally recommended approach to communicate between Verilog modules is through **port connections** within a higher-level module. Think of it like connecting pre-made circuit boards; the ports on each board correspond to pins on connectors, establishing the physical link for data transfer. Each module has its own input and output ports, which are declared within its `module` definition. These ports are then connected in a top-level module, often referred to as a testbench or a higher-level design module. This method enables direct, synchronous communication where data is typically transferred on the rising or falling edge of a clock signal. The directionality of signals must be accurately defined (input or output), ensuring that data is sourced from an output port and received by a corresponding input port. The data type also must match across the connecting ports to avoid synthesis errors.

While direct port connections are ideal for simple interfaces, more complex designs often require more structured communication methods. For example, when dealing with shared resources or when timing uncertainties make synchronous transfer impractical, alternative strategies like using **FIFOs (First-In, First-Out buffers)** or **handshaking protocols** become vital. FIFOs act as intermediate buffers, accommodating rate differences between modules. A module may be processing data faster than it is consumed, and a FIFO will store the data until the consuming module is ready. Conversely, handshaking protocols establish a more formal communication flow, involving request and acknowledgment signals, providing robust synchronization particularly when modules operate on different clock domains.

Additionally, one can utilize **buses** to communicate between multiple modules when sharing common data. A bus is a collection of wires that carry related information simultaneously. Bus architectures can be addressable and can be used to connect multiple modules to a shared memory location or to provide a wider data path. Proper use of address decoders, and careful management of bus contention is essential for robust operation.

Let's consider three code examples to illustrate different techniques:

**Example 1: Direct Port Connection - A simple adder module**

```verilog
module adder (
  input  wire [7:0] a,
  input  wire [7:0] b,
  output wire [7:0] sum
);

  assign sum = a + b;

endmodule

module top_level_adder(
   input  wire [7:0] input_a,
   input  wire [7:0] input_b,
   output wire [7:0] output_sum
);

  wire [7:0] intermediate_sum;

  adder adder_instance(
    .a(input_a),
    .b(input_b),
    .sum(intermediate_sum)
    );

  assign output_sum = intermediate_sum;

endmodule
```
In this example, the `adder` module takes two 8-bit inputs (`a` and `b`) and outputs their sum (`sum`). The `top_level_adder` instantiates the `adder` and connects it's input ports directly to the input ports of the `top_level_adder` and the output of the `adder` is routed to the output of the `top_level_adder`. Here, the connection is straightforward; the signals are directly passed between the module ports. This is the simplest form of communication and is suited for basic functional blocks. Note the use of `wire` variables `intermediate_sum`, necessary when connecting modules.

**Example 2: Using a FIFO for asynchronous communication**

```verilog
module fifo #(parameter DATA_WIDTH = 8, parameter DEPTH = 16) (
  input  wire             clk,
  input  wire             wr_en,
  input  wire [DATA_WIDTH-1:0] wr_data,
  output wire             full,
  input  wire             rd_en,
  output wire [DATA_WIDTH-1:0] rd_data,
  output wire             empty
);

  reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];
  reg [$clog2(DEPTH)-1:0] wr_ptr;
  reg [$clog2(DEPTH)-1:0] rd_ptr;
  reg  [DEPTH-1:0]  fill_level;

  assign full = (fill_level == DEPTH) ? 1'b1 : 1'b0;
  assign empty = (fill_level == 0) ? 1'b1 : 1'b0;


  always @(posedge clk) begin
    if (wr_en && !full) begin
      mem[wr_ptr] <= wr_data;
      wr_ptr <= wr_ptr + 1;
        fill_level <= fill_level + 1;
    end
    if (rd_en && !empty) begin
        rd_data <= mem[rd_ptr];
      rd_ptr <= rd_ptr + 1;
      fill_level <= fill_level - 1;
    end
  end


endmodule


module producer (
   input   wire clk,
   output  wire data_ready,
   output  wire [7:0] data_out,
   input   wire fifo_full
  );
  reg [7:0] counter;
  assign data_ready = !fifo_full;

  always @(posedge clk)
  begin
    if (data_ready) begin
      counter <= counter + 1;
      data_out <= counter;
    end
  end
endmodule

module consumer (
    input   wire clk,
    input   wire data_valid,
    input   wire [7:0] data_in
);
    reg [7:0] received_data;

    always @(posedge clk) begin
        if(data_valid)
            received_data <= data_in;
    end
endmodule


module top_level_fifo(
    input wire clk,
    output wire [7:0] consumed_data
    );

    wire        fifo_full;
    wire [7:0]  producer_data;
    wire        data_ready;
    wire        fifo_empty;
    wire [7:0]  fifo_data_out;

    producer producer_instance(
        .clk(clk),
        .data_ready(data_ready),
        .data_out(producer_data),
        .fifo_full(fifo_full)
    );

    fifo #(
      .DATA_WIDTH(8),
      .DEPTH(16)
      ) fifo_instance(
      .clk(clk),
      .wr_en(data_ready),
      .wr_data(producer_data),
      .full(fifo_full),
      .rd_en(!fifo_empty),
      .rd_data(fifo_data_out),
      .empty(fifo_empty)
      );

    consumer consumer_instance(
      .clk(clk),
      .data_valid(!fifo_empty),
      .data_in(fifo_data_out)
    );

    assign consumed_data = fifo_data_out;

endmodule

```
This example demonstrates a parameterized `fifo` module, which includes flags indicating full and empty states. It uses registers to store data, write and read pointers, and uses `always` blocks to handle write and read operations. The `producer` module generates data when the FIFO is not full, and the `consumer` module reads data when the FIFO is not empty. A top level module instantiates these three modules and connects their relevant ports. This is more complex than the previous example but essential when dealing with rate mismatches between modules.

**Example 3: Handshaking Protocol for reliable asynchronous transfer**

```verilog

module sender (
  input  wire             clk,
  input  wire [7:0]       data_in,
  input  wire             ready,
  output wire             valid,
  output wire [7:0]       data_out
);

    reg valid_r;
    reg [7:0] data_out_r;

    always @(posedge clk) begin
      if (ready) begin
        valid_r <= 1'b1;
        data_out_r <= data_in;
      end else begin
        valid_r <= 1'b0;
      end
    end

    assign valid = valid_r;
    assign data_out = data_out_r;


endmodule

module receiver (
  input  wire clk,
  input  wire             valid,
  input  wire [7:0]       data_in,
  output wire             ready,
  output wire [7:0]      data_out
);
    reg ready_r;
    reg [7:0] data_out_r;

    always @(posedge clk)
    begin
      if(valid) begin
          ready_r <= 1'b0;
          data_out_r <= data_in;
      end else begin
          ready_r <= 1'b1;
      end
    end

    assign ready = ready_r;
    assign data_out = data_out_r;
endmodule


module top_level_handshake (
    input  wire clk,
    output wire [7:0]    received_data
  );

    wire data_valid;
    wire data_ready;
    wire [7:0] sender_data_out;
    wire [7:0] receiver_data_out;
    reg [7:0]  data_to_send;

    always @(posedge clk) begin
      data_to_send <= data_to_send + 1;
    end

    sender sender_instance (
      .clk(clk),
      .data_in(data_to_send),
      .ready(data_ready),
      .valid(data_valid),
      .data_out(sender_data_out)
    );

    receiver receiver_instance (
      .clk(clk),
      .valid(data_valid),
      .data_in(sender_data_out),
      .ready(data_ready),
      .data_out(receiver_data_out)
    );

    assign received_data = receiver_data_out;

endmodule
```
In this third example, we have modules communicating using a handshaking protocol. The `sender` module asserts a `valid` signal when data is available, and the `receiver` module asserts a `ready` signal when it is prepared to receive data. A top level module instantiates these two modules and connects their relevant ports. This is crucial for reliable data transfer in situations where synchronous communication is not feasible. This is a classic example of an interface, demonstrating synchronous data transfer on different clock domains. The protocol is commonly called a valid and ready interface.

These examples highlight several ways to connect Verilog modules. Each serves a particular purpose and has advantages based on the application. For further study, consider reviewing material on:

*   **Digital Design Fundamentals**: Understanding basic building blocks like multiplexers, decoders, and registers will provide a solid foundation.
*   **Synchronous vs. Asynchronous Design**: This knowledge is crucial for choosing the right communication method between modules.
*   **Bus Architectures**: Exploring different bus topologies and protocols is essential for handling complex data communication in larger designs.
*   **Verification Techniques**: Knowing how to write testbenches that thoroughly exercise communication protocols is crucial to catch errors early.

By mastering these techniques, you'll be well-equipped to develop complex and robust Verilog designs.
