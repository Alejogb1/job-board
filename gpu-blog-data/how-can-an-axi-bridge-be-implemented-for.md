---
title: "How can an AXI bridge be implemented for HPS-to-FPGA communication?"
date: "2025-01-30"
id: "how-can-an-axi-bridge-be-implemented-for"
---
The fundamental challenge in implementing an AXI bridge for High Performance System (HPS) to Field-Programmable Gate Array (FPGA) communication lies in the impedance mismatch and protocol differences between the HPS's memory-mapped world and the FPGA's programmable logic. Typically, the HPS operates with high-speed, tightly coupled memory interfaces while the FPGA logic needs a flexible yet standardized interface for custom hardware acceleration. I have encountered this issue multiple times when developing custom System-on-Chip (SoC) solutions and have found that a well-defined AXI bridge serves as the essential translator.

An AXI bridge facilitates communication by abstracting the complexity of the AXI protocol, allowing the HPS to interact with FPGA peripherals as if they were simply memory locations. This involves translating HPS-generated addresses, data, and control signals into the appropriate AXI transactions that the FPGA-based peripherals can understand. Conversely, responses from the FPGA need to be translated back to the HPS. This bi-directional conversion is the core function of the AXI bridge.

The specific steps in creating such a bridge revolve around several key elements: mastering the AXI protocol; selecting a suitable interface within the FPGA; ensuring timing constraints are met; and developing robust control logic. I typically approach this problem by first identifying the precise AXI bus width (e.g., 32-bit, 64-bit) required based on the performance needs of the peripherals to be accessed. The HPS's memory map dictates the base addresses for the regions where these peripherals will be mapped, requiring careful planning.

Let's consider a simplified scenario where the HPS wants to read and write data to an FPGA-based FIFO. I will illustrate through code, using a high-level pseudo-HDL language suitable for FPGA development; consider it a close derivative of Verilog, but optimized for illustrative purposes:

**Code Example 1: Basic AXI Slave Interface (Write)**

```pseudo-hdl
module axi_slave_write (
    input  wire             clk,
    input  wire             reset,
    input  wire [31:0]      s_axi_awaddr,   // AXI Write Address
    input  wire [3:0]       s_axi_awprot,   // AXI Write Protection
    input  wire             s_axi_awvalid,  // AXI Write Address Valid
    output reg              s_axi_awready,  // AXI Write Address Ready
    input  wire [31:0]      s_axi_wdata,    // AXI Write Data
    input  wire [3:0]       s_axi_wstrb,    // AXI Write Strobe
    input  wire             s_axi_wvalid,   // AXI Write Data Valid
    output reg              s_axi_wready,   // AXI Write Data Ready
    output reg              s_axi_bvalid,   // AXI Write Response Valid
    input  wire              s_axi_bready,   // AXI Write Response Ready
    output reg [1:0]       s_axi_bresp,    // AXI Write Response

    output reg [31:0]       fifo_wdata,     // Data to FIFO
    output reg              fifo_wren      // FIFO Write Enable
);
  reg [31:0]  local_addr;
  reg         write_in_progress;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
        s_axi_awready <= 0;
        s_axi_wready  <= 0;
        s_axi_bvalid  <= 0;
        write_in_progress <= 0;
        fifo_wren     <= 0;
        s_axi_bresp   <= 2'b00; //OKAY
    end else begin
       if (s_axi_awvalid && !write_in_progress) begin
          s_axi_awready  <= 1;
          local_addr     <= s_axi_awaddr;
          write_in_progress <= 1;
       end else begin
          s_axi_awready <= 0;
       end

       if (s_axi_wvalid && write_in_progress) begin
          s_axi_wready <= 1;
          fifo_wdata <= s_axi_wdata;
          fifo_wren  <= 1;
          s_axi_bvalid  <= 1;
          s_axi_bresp    <= 2'b00; //OKAY

       end else begin
          s_axi_wready <= 0;
          fifo_wren <= 0;
          s_axi_bvalid <= 0;
       end

       if (s_axi_bvalid && s_axi_bready) begin
          write_in_progress <= 0;
       end
    end
  end
endmodule
```
This module demonstrates a simplified AXI slave interface for write operations. It highlights the handshake logic: `s_axi_awvalid`/`s_axi_awready` for address, `s_axi_wvalid`/`s_axi_wready` for data, and `s_axi_bvalid`/`s_axi_bready` for the write response. The key takeaway is how the `s_axi_wdata` is directly mapped to `fifo_wdata` along with a write enable (`fifo_wren`). Note that this implementation assumes a fixed-address space and does not include any complex address decoding or burst handling for simplicity. The `s_axi_bresp` signals a successful transaction.

**Code Example 2: Basic AXI Slave Interface (Read)**

```pseudo-hdl
module axi_slave_read (
    input  wire             clk,
    input  wire             reset,
    input  wire [31:0]      s_axi_araddr,   // AXI Read Address
    input  wire [3:0]       s_axi_arprot,   // AXI Read Protection
    input  wire             s_axi_arvalid,  // AXI Read Address Valid
    output reg              s_axi_arready,  // AXI Read Address Ready
    output reg [31:0]      s_axi_rdata,    // AXI Read Data
    output reg [1:0]       s_axi_rresp,    // AXI Read Response
    output reg              s_axi_rvalid,   // AXI Read Data Valid
    input  wire             s_axi_rready,   // AXI Read Data Ready

    input  wire [31:0]      fifo_rdata,     // Data from FIFO
    input  wire             fifo_rdreq      // FIFO Read Enable
);
  reg         read_in_progress;
  reg [31:0] local_addr;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      s_axi_arready <= 0;
      s_axi_rdata   <= 0;
      s_axi_rresp   <= 2'b00; //OKAY
      s_axi_rvalid  <= 0;
      read_in_progress <= 0;
      fifo_rdreq    <= 0;
    end else begin
       if (s_axi_arvalid && !read_in_progress) begin
           s_axi_arready  <= 1;
           local_addr      <= s_axi_araddr;
           read_in_progress <= 1;
           fifo_rdreq       <= 1; //Start FIFO read
        end else begin
            s_axi_arready <= 0;
            fifo_rdreq    <= 0;
        end

        if(read_in_progress) begin
          s_axi_rdata  <= fifo_rdata;
          s_axi_rresp   <= 2'b00; //OKAY
          s_axi_rvalid <= 1;
        end else begin
          s_axi_rvalid <= 0;
        end

       if(s_axi_rvalid && s_axi_rready) begin
          read_in_progress <= 0;
       end
    end
  end
endmodule
```

The read operation is similar to write, but involves the `s_axi_ar` signals for address and `s_axi_r` signals for data. Note how data from the `fifo_rdata` is directly driven onto `s_axi_rdata` after the read request. The `fifo_rdreq` is asserted to request data from the FIFO.

**Code Example 3: Top-Level Instantiation**

```pseudo-hdl
module top_level (
    input   wire             clk,
    input   wire             reset,
    // AXI Interface from HPS
    input  wire [31:0]      s_axi_awaddr,
    input  wire [3:0]       s_axi_awprot,
    input  wire             s_axi_awvalid,
    output wire             s_axi_awready,
    input  wire [31:0]      s_axi_wdata,
    input  wire [3:0]       s_axi_wstrb,
    input  wire             s_axi_wvalid,
    output wire             s_axi_wready,
    output wire             s_axi_bvalid,
    input  wire             s_axi_bready,
    output wire [1:0]       s_axi_bresp,

    input  wire [31:0]      s_axi_araddr,
    input  wire [3:0]       s_axi_arprot,
    input  wire             s_axi_arvalid,
    output wire             s_axi_arready,
    output wire [31:0]      s_axi_rdata,
    output wire [1:0]       s_axi_rresp,
    output wire             s_axi_rvalid,
    input  wire             s_axi_rready,
    // FIFO Interface
    input wire  [31:0] fifo_rdata,
    input wire            fifo_full,
    input wire             fifo_empty,
    output wire [31:0]     fifo_wdata,
    output wire            fifo_wren,
    output wire            fifo_rdreq
);

  axi_slave_write axi_write_inst (
    .clk(clk),
    .reset(reset),
    .s_axi_awaddr(s_axi_awaddr),
    .s_axi_awprot(s_axi_awprot),
    .s_axi_awvalid(s_axi_awvalid),
    .s_axi_awready(s_axi_awready),
    .s_axi_wdata(s_axi_wdata),
    .s_axi_wstrb(s_axi_wstrb),
    .s_axi_wvalid(s_axi_wvalid),
    .s_axi_wready(s_axi_wready),
    .s_axi_bvalid(s_axi_bvalid),
    .s_axi_bready(s_axi_bready),
    .s_axi_bresp(s_axi_bresp),
    .fifo_wdata(fifo_wdata),
    .fifo_wren(fifo_wren)
  );

  axi_slave_read axi_read_inst (
    .clk(clk),
    .reset(reset),
    .s_axi_araddr(s_axi_araddr),
    .s_axi_arprot(s_axi_arprot),
    .s_axi_arvalid(s_axi_arvalid),
    .s_axi_arready(s_axi_arready),
    .s_axi_rdata(s_axi_rdata),
    .s_axi_rresp(s_axi_rresp),
    .s_axi_rvalid(s_axi_rvalid),
    .s_axi_rready(s_axi_rready),
    .fifo_rdata(fifo_rdata),
    .fifo_rdreq(fifo_rdreq)
  );


endmodule
```

This module instantiates both the write and read slave modules, connecting them to the main AXI interface and the FIFO interface. It shows how the individual modules are integrated into a top-level design. In a real design, address decoding within the `top_level` module would determine which slave instance is active. In the pseudo-code, this decoding is omitted for clarity.

These code examples highlight the core components of a basic AXI slave implementation suitable for simple HPS-to-FPGA interaction. However, real-world implementations will be substantially more complex, encompassing handling for burst transactions, address decoding for multiple peripherals, and error handling. Furthermore, timing closure is critical and must be carefully addressed during implementation. In my experience, the timing analysis tools are vital for ensuring the design meets performance requirements.

To further enhance your understanding and implement practical AXI bridges, I recommend exploring the following resources. First, consult the official AMBA AXI protocol specification document, which outlines the exact timing and signal requirements. Second, work through example projects and tutorials found in FPGA vendor documentation (Intel, Xilinx). These resources often provide step-by-step guides for integrating AXI peripherals into their development environments. Finally, examine academic papers and conference proceedings that explore advanced techniques in AXI bridge design and optimization. Through a blend of these resources and hands-on experience, the ability to create reliable AXI bridges for HPS-to-FPGA communication can be developed.
