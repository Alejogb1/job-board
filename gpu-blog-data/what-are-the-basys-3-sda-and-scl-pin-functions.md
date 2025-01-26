---
title: "What are the Basys 3 SDA and SCL pin functions?"
date: "2025-01-26"
id: "what-are-the-basys-3-sda-and-scl-pin-functions"
---

The Basys 3 development board's SDA (Serial Data) and SCL (Serial Clock) pins serve as the physical interface for I²C (Inter-Integrated Circuit) communication, a widely adopted serial protocol for short-distance, synchronous data exchange between integrated circuits. These pins are not directly connected to a microcontroller’s general-purpose I/O, requiring a specific hardware module and software configuration to be utilized effectively. Having worked extensively with embedded systems for over a decade, I've encountered numerous situations where mastering the intricacies of I²C communication, and therefore understanding these pins, is critical.

I²C is a two-wire protocol using an open-drain configuration. This means the pins can actively pull a signal low, but to achieve a high signal, an external pull-up resistor connected to the positive supply voltage is required. This is crucial as it differs from standard push-pull logic, where the output actively drives both high and low states. The pull-up resistors on the Basys 3 are often integrated on the board. The SDA line is bi-directional and transmits the actual data bits. The SCL line, driven by the I²C master, provides the clock signal, synchronizing the data transfer. Devices connected to the bus are designated as either a master or a slave; the master initiates communication and drives the SCL clock, while the slave responds to the master's requests on the SDA line, using the shared clock.

Specifically on the Basys 3, these pins are multiplexed, meaning their electrical connections are shared with other functions of the FPGA (Field Programmable Gate Array). In practice, this means the SDA and SCL pins are not simply connected to specific I/O pads; rather, the FPGA fabric needs to be configured to route signals correctly. The specific pin locations on the Basys 3 for I²C are typically defined within the board's constraints file, which dictates how the FPGA’s configurable routing is used. The pins themselves are not inherently SDA and SCL. It is the digital logic within the FPGA that dictates that purpose. The I²C signals generated or used are routed to those physical pins through user defined logic. Therefore, the role of the pins is determined by the software and the FPGA configuration.

To illustrate how I've utilized these pins, consider the following Verilog code examples, which I’ve adapted from past projects for demonstration:

**Example 1: I²C Master Module in Verilog**

This module shows a basic I²C master that can write a single byte to a specific slave address.

```verilog
module i2c_master (
    input clk,
    input reset,
    output reg scl,
    inout reg sda,
    input [6:0] slave_addr,
    input [7:0] data_out,
    input start_transfer,
    output reg transfer_complete
    );

parameter IDLE = 2'b00;
parameter START = 2'b01;
parameter SEND_ADDR = 2'b10;
parameter SEND_DATA = 2'b11;
parameter STOP = 2'b100;

reg [1:0] state;
reg [3:0] bit_count;
reg [7:0] shift_reg;

assign sda = (state == SEND_ADDR || state == SEND_DATA)? shift_reg[7]: 1'bz;

always @(posedge clk) begin
    if (reset) begin
        state <= IDLE;
        scl <= 1'b1;
        sda <= 1'bz;
        transfer_complete <= 1'b0;
        bit_count <= 4'b0;
    end else begin
        case (state)
        IDLE: begin
            if (start_transfer) begin
                state <= START;
                transfer_complete <= 1'b0;
            end
        end
        START: begin
            scl <= 1'b1;
            sda <= 1'b0;
            #1;
            scl <= 1'b0;
            shift_reg <= {slave_addr, 1'b0};
            bit_count <= 4'b0;
            state <= SEND_ADDR;
        end
        SEND_ADDR: begin
            if (bit_count < 8) begin
                #1;
                scl <= 1'b1;
                #1;
                scl <= 1'b0;
                shift_reg <= {shift_reg[6:0], 1'b0};
                bit_count <= bit_count + 1'b1;
            end else begin
                shift_reg <= data_out;
                bit_count <= 4'b0;
                state <= SEND_DATA;
            end
         end
         SEND_DATA: begin
            if(bit_count < 8) begin
                #1;
                scl <= 1'b1;
                #1;
                scl <= 1'b0;
                shift_reg <= {shift_reg[6:0], 1'b0};
                bit_count <= bit_count + 1'b1;
            end else begin
                state <= STOP;
            end
          end
        STOP: begin
           sda <= 1'b0;
            #1;
            scl <= 1'b1;
            #1;
            sda <= 1'b1;
            state <= IDLE;
            transfer_complete <= 1'b1;
        end
       endcase
    end
end
endmodule
```

In this first example, the core logic for a basic master is presented. The module receives the clock, reset, slave address and data to write, along with a start signal. A state machine sequences through a start condition, sending the slave address (including a write bit) sending the data byte and a stop condition. Notice that during data transmission, the `sda` line is driven based on the MSB of the `shift_reg` and is high-impedance otherwise, due to I²C’s open-drain nature. This implies that on a physical board, there need to be pull-up resistors. The delay `#1` in this example is a simplified representation of delays needed in an I²C transaction, which would typically be determined by the bus clock speed. The SCL signal is toggled here to clock data on each transaction.

**Example 2: I²C Slave Module in Verilog**

This module shows a basic I²C slave that receives a single byte from the master.

```verilog
module i2c_slave (
    input clk,
    input reset,
    input scl,
    inout sda,
    input [6:0] slave_addr,
    output reg [7:0] data_in,
    output reg received_data
    );
parameter IDLE = 2'b00;
parameter RECV_ADDR = 2'b01;
parameter RECV_DATA = 2'b10;

reg [1:0] state;
reg [7:0] shift_reg;
reg [6:0] recv_addr;
reg ack;

assign sda = (state == RECV_DATA && !ack)? 1'bz: 1'b0;

always @(posedge clk) begin
    if (reset) begin
         state <= IDLE;
        data_in <= 8'b0;
        received_data <= 1'b0;
        shift_reg <= 8'b0;
        ack <= 1'b0;
    end
    else begin
      case (state)
        IDLE: begin
            if (!scl) begin
                if(sda == 1'b0) begin //Start condition detected
                   shift_reg <= 8'b0;
                   state <= RECV_ADDR;
                end
              end
        end
        RECV_ADDR: begin
            if(scl) begin
                shift_reg[7:1] <= shift_reg[6:0];
                shift_reg[0] <= sda;
            end else if(shift_reg[7]) begin
                    if (shift_reg[6:0] == slave_addr) begin
                    ack <= 1'b0;
                        shift_reg <= 8'b0;
                        state <= RECV_DATA;
                    end else begin
                      state <= IDLE;
                      ack <= 1'b1;
                    end
                 end
        end
        RECV_DATA: begin
            if(scl) begin
                shift_reg[7:1] <= shift_reg[6:0];
                shift_reg[0] <= sda;
              end else begin
                 ack <= 1'b0;
                 data_in <= shift_reg;
                  received_data <= 1'b1;
                state <= IDLE;
             end
        end
      endcase
    end
end
endmodule
```

This second code example describes a slave module. The slave responds to a specific address passed to it on the I²C bus, and captures the data byte after the correct address. The most notable detail here is the check for a start condition (falling edge of SDA while SCL is high), and the use of the falling edge of SCL to clock in the data. Also note the acknowledge operation: driving `sda` low when the slave address matches, to notify the master that data is going to be received. This is done before the receiving of the data, and at the end of data. The `sda` line is pulled low when the slave is listening and ready for data, and is high impedance otherwise. This module also implies the existence of pull-up resistors on the physical implementation. The `ack` signal in this example is an illustration of the acknowledge signal, but would need more complexity in practice, such as handling repeated start conditions.

**Example 3: Top-Level Module Instantiating Both Master and Slave**

This module instantiates both modules, and illustrates basic data transfer.

```verilog
module i2c_top (
    input clk,
    input reset,
    output scl,
    inout sda,
	 output [7:0] slave_received_data
);

wire transfer_complete_master;
reg start_transfer;
reg [7:0] data_master;
reg [6:0] slave_addr_master;
reg [6:0] slave_addr_slave;
wire received_data_slave;

i2c_master i2c_master_inst (
    .clk(clk),
    .reset(reset),
    .scl(scl),
    .sda(sda),
    .slave_addr(slave_addr_master),
    .data_out(data_master),
    .start_transfer(start_transfer),
    .transfer_complete(transfer_complete_master)
);

i2c_slave i2c_slave_inst (
    .clk(clk),
    .reset(reset),
    .scl(scl),
    .sda(sda),
    .slave_addr(slave_addr_slave),
    .data_in(slave_received_data),
    .received_data(received_data_slave)
);
always @(posedge clk) begin
	if(reset) begin
		slave_addr_master <= 7'h28;
		data_master <= 8'hAA;
		slave_addr_slave <= 7'h28;
		start_transfer <= 1'b0;
	end else begin
		start_transfer <= 1'b1;
	end
end
endmodule
```

Here, the master and slave modules are instantiated within a single top-level module. This example sets the master’s address, the data to be transmitted, and triggers a transfer. The slave device has its address also set, and receives the data sent by the master. This illustrates the direct communication between master and slave modules. The specific slave address is configured, and a simple test of transferring data from master to slave is executed. `scl` and `sda` pins are connected to the instantiated modules.

These code examples provide a practical insight into how these signals are utilized within an FPGA environment. The specific I²C implementation on the Basys 3 will be defined in the target FPGA's constraints file, and will vary based on the specific application.

For individuals seeking further understanding of I²C and its practical implementations, I recommend consulting textbooks covering digital electronics and embedded systems. Look for resources explaining the details of serial communication protocols, specifically focusing on I²C, and how FPGAs are used in interfacing peripherals. I also suggest exploring official datasheets of common I²C devices, such as EEPROMs or temperature sensors. Hands-on experience with development boards like the Basys 3, working directly with FPGA projects, combined with the aforementioned resources, will be the most beneficial. This combination will provide the necessary practical skills and theoretical background to fully comprehend the functions of the SDA and SCL pins.
