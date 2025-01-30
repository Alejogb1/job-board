---
title: "How can I convert an ADC's serial output to an N-bit parallel signal?"
date: "2025-01-30"
id: "how-can-i-convert-an-adcs-serial-output"
---
Analog-to-digital converters (ADCs) often output data serially to minimize the number of pins required, especially at higher resolutions and sampling rates.  Converting this serial output into a parallel N-bit word is a common requirement for interfacing with digital logic, processing units, or memory. My experience designing embedded systems has frequently involved this, and while the specific implementation varies based on the ADC's serial protocol, the fundamental principles remain consistent.

The conversion process essentially reverses the serialization process within the ADC. Instead of transmitting bits sequentially, the system collects them sequentially into a register and then presents all bits at once as a parallel output.  The key steps involve identifying the start of a new conversion cycle, clocking in the serial data stream, and reassembling it into the desired parallel format. This often requires a combination of shift registers, counters, and some control logic. Here's how I generally approach this problem:

The first challenge is understanding the ADC's specific serial protocol. Protocols such as SPI (Serial Peripheral Interface), I2C (Inter-Integrated Circuit), or proprietary formats dictate how data is synchronized and transmitted.  SPI is especially common in high-speed ADCs, utilizing a clock signal (SCK) for synchronization, a data output line (SDO), and a chip select signal (CS) to activate a specific device on the bus.  I2C, while also serial, is typically used for lower-speed applications and requires more complex handling on the master.  A thorough datasheet review is vital to determine data timing, framing, and bit order. The data sheet will detail if the MSB is sent first or LSB, which is paramount in our conversion.

Once the protocol is understood, we can implement a state machine in hardware using a Field-Programmable Gate Array (FPGA) or a dedicated microcontroller.  Alternatively, if speed requirements are relaxed, software handling on a microcontroller could also suffice. My preference leans towards hardware implementations for higher performance systems.

Let’s illustrate with three examples of increasing complexity, focused on SPI since it’s pervasive in this area.  We’ll assume a generic ADC with an SPI interface and an 8-bit output for the first example.

**Example 1: Basic 8-bit Parallel Conversion with Minimal Logic**

In the simplest scenario, where timing is non-critical and the clock speed of the SPI interface is slow, we can use a basic shift register and load it directly with the incoming data. The core idea is to clock the serial data into a shift register and, once complete, latch the register's contents into a parallel output. In an FPGA, the verilog to do this would look something like this:

```verilog
module spi_to_parallel_8bit (
    input clk, // System clock
    input spi_clk, // SPI clock
    input spi_data, // SPI data line (SDO)
    input spi_cs_n, // SPI chip select, active low
    output [7:0] parallel_data,
    output data_valid
);

reg [7:0] shift_reg;
reg [3:0] bit_counter;
reg data_valid_reg;

always @(posedge clk) begin
    if (!spi_cs_n) begin // Chip select is low, start of transmission
        if (spi_clk) begin
           shift_reg <= {shift_reg[6:0], spi_data};
           bit_counter <= bit_counter + 1;
        end
        if(bit_counter == 7)begin // 8 bits received
            data_valid_reg <= 1;
        end
    end else begin
      bit_counter <= 0;
      data_valid_reg <= 0;
    end
end

assign parallel_data = shift_reg;
assign data_valid = data_valid_reg;

endmodule
```

This Verilog module, while simplified, illustrates the process. When `spi_cs_n` is low (active low), the module registers data on each positive edge of the SPI clock (`spi_clk`). After 8 clocks, `bit_counter` increments and `data_valid_reg` signals that `parallel_data` has the complete 8 bits. When chip select (`spi_cs_n`) goes high the internal registers are reset. The key here is synchronizing with the `spi_clk` and ensuring it is not too fast for our internal clock `clk`. Note, this module assumes that our internal clock is higher than the frequency of `spi_clk` and the bit order is MSB first. If LSB first, the shift operation would need to shift the LSB bit into index 7 of the register (`shift_reg <= {spi_data, shift_reg[7:1]}`), this is common in SPI.

**Example 2: Handling Variable N-bit Output with a Counter**

Now, let’s consider a scenario where the ADC resolution is configurable, or we simply require a larger number of bits. A simple shift register will no longer cut it. We’ll need a counter to keep track of the number of bits shifted and a larger register. This could be the case for ADCs with 12-bit or 16-bit outputs. Again, implementing this in verilog would be something like:

```verilog
module spi_to_parallel_nbit (
    input clk,  // System clock
    input spi_clk, // SPI clock
    input spi_data, // SPI data line (SDO)
    input spi_cs_n, // SPI chip select, active low
    parameter DATA_WIDTH = 16, // configurable output width
    output [DATA_WIDTH-1:0] parallel_data,
    output data_valid
);

reg [DATA_WIDTH-1:0] shift_reg;
reg [4:0] bit_counter;
reg data_valid_reg;

always @(posedge clk) begin
    if (!spi_cs_n) begin // Chip select is low, start of transmission
        if (spi_clk) begin
           shift_reg <= {shift_reg[DATA_WIDTH-2:0], spi_data};
           bit_counter <= bit_counter + 1;
        end
         if(bit_counter == (DATA_WIDTH -1))begin // DATA_WIDTH bits received
            data_valid_reg <= 1;
        end
    end else begin
        bit_counter <= 0;
        data_valid_reg <= 0;
    end
end

assign parallel_data = shift_reg;
assign data_valid = data_valid_reg;
endmodule
```

The most important change here is the introduction of the parameter `DATA_WIDTH`. This allows us to easily change the bit width of the parallel output without modifying the logic. The counter `bit_counter` keeps track of how many bits we've shifted in, and `data_valid` only goes high once we have completed a full conversion. Again, this assumes MSB first and the internal clock is faster than `spi_clk`.

**Example 3:  Double Buffering for Continuous Data Acquisition**

For applications requiring continuous data acquisition, directly latching the shift register’s output could lead to timing issues, especially when the processing of data takes more time than the ADC conversion time. I often use double buffering in these cases.  This involves two parallel registers: one that is being loaded with the serial data from the ADC and another which provides a static output. Once a complete conversion is in the input register, the data is copied to the output register which is then used for further processing. While the data is being processed, we can simultaneously load the next conversion. Let’s add another register, and a select signal to switch between them.

```verilog
module spi_to_parallel_double_buffer (
    input clk, // System clock
    input spi_clk,  // SPI clock
    input spi_data,  // SPI data line (SDO)
    input spi_cs_n,  // SPI chip select, active low
    parameter DATA_WIDTH = 16, // configurable output width
    output [DATA_WIDTH-1:0] parallel_data,
    output data_valid
);

reg [DATA_WIDTH-1:0] shift_reg_a;
reg [DATA_WIDTH-1:0] shift_reg_b;
reg [DATA_WIDTH-1:0] active_reg;
reg [4:0] bit_counter;
reg data_valid_reg;
reg buffer_select;


always @(posedge clk) begin
   if (!spi_cs_n) begin // Chip select is low, start of transmission
      if(spi_clk) begin
         if(!buffer_select) begin
             shift_reg_a <= {shift_reg_a[DATA_WIDTH-2:0], spi_data};
         end else begin
             shift_reg_b <= {shift_reg_b[DATA_WIDTH-2:0], spi_data};
         end
         bit_counter <= bit_counter + 1;
      end
        if(bit_counter == (DATA_WIDTH-1)) begin // DATA_WIDTH bits received
            data_valid_reg <= 1;
            buffer_select <= ~buffer_select;
           if(!buffer_select)
                active_reg <= shift_reg_b;
           else
                active_reg <= shift_reg_a;

        end
   end else begin
        bit_counter <= 0;
        data_valid_reg <= 0;
    end
end

assign parallel_data = active_reg;
assign data_valid = data_valid_reg;
endmodule
```

Here, we introduce two shift registers: `shift_reg_a` and `shift_reg_b`. The `buffer_select` register toggles after each conversion cycle. When `buffer_select` is 0, the serial data is clocked into `shift_reg_a`, and when it's 1, the data goes into `shift_reg_b`. After the conversion, the completed conversion is moved into `active_reg` before the register begins to take more data. This creates a buffer so data is always available.

For further learning, I recommend delving deeper into specific ADC datasheets to understand the nuanced timing requirements of different protocols. Books such as “Digital Design” by Morris Mano, and texts focusing on FPGA design with Verilog or VHDL provide a strong theoretical background. For a more practical approach to embedded systems, resources focused on microcontroller programming and interfacing can be invaluable. Finally, examining reference designs published by semiconductor manufacturers for your specific ADC device can offer valuable insight into best practices.
