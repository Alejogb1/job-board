---
title: "Why does the Verilog UART have a long convergence time?"
date: "2025-01-30"
id: "why-does-the-verilog-uart-have-a-long"
---
In my experience designing custom ASICs for embedded systems, I’ve repeatedly encountered the challenge of achieving rapid convergence with UART implementations in Verilog, and the reasons are multifaceted. A significant contributor to extended convergence times, particularly during simulation, lies within the inherent asynchronous nature of the UART protocol coupled with how digital hardware description languages simulate time. The interplay between the data rate, clock frequency, and simulation time scales can drastically impact the observed time until the receiver reliably locks onto the incoming bit stream. This problem manifests itself during initial synchronization or when significant clock skews are introduced.

The UART, fundamentally, relies on asynchronous communication, meaning it does not utilize a shared clock signal between transmitter and receiver. Instead, the receiver must sample the incoming data stream at a rate derived from its internal clock, using the start bit’s falling edge to synchronize the sampling process for the following data and stop bits. Ideally, the receiver’s sampling clock should be close to the transmitter’s bit rate. However, perfect synchronization is rare, leading to oversampling at several times the bit rate to compensate for rate mismatches and noise. This oversampling adds inherent latency. While oversampling aids in robust data recovery, it also means the receiver spends several clock cycles observing each bit, which contributes to perceived convergence delay.

The convergence time in Verilog simulations often appears longer than real-world hardware because of the nature of discrete time steps within the simulator. Each simulation tick represents a defined increment of time, usually significantly smaller than the period of the system clock. The simulation process emulates the state transitions of the digital logic over these discrete time steps. During the asynchronous UART start bit detection, the exact point at which the falling edge of start bit occurs in relation to the receiver’s clock is somewhat ambiguous within the digital domain. If the simulation’s time resolution is not fine enough, the initial detection of this start bit can be delayed by up to a full clock cycle in the simulation environment, leading to an initial perceived delay before the sampling process begins. Subsequent data bit sampling is also discretized, which results in a further accumulation of small delays, making the overall synchronization process seem considerably slower. In a real hardware implementation, this start bit detection is more instantaneous and occurs according to the physical properties of the circuit elements.

Additionally, the handling of the receiver’s edge detection logic in Verilog plays a crucial role. A simple edge detector might employ a two-register stage to compare the current input signal to the previous state. This introduces two clock cycle delays, one for storage and one for the comparison. These delays are a crucial aspect of signal synchronization when going from an asynchronous external signal to a synchronous system. The delay required to capture and then correctly register the start bit, plus the latency in the sampling of subsequent data, all contribute to the overall convergence time and are far more pronounced in simulation. This is because simulation operates on a precise tick-based system that perfectly tracks every signal state, whereas a physical implementation of the UART is subject to analog noise and propagation delays, which can mask the discrete, step-by-step behavior inherent in a simulator.

Let's examine some illustrative Verilog code examples that demonstrate these challenges:

**Example 1: Start bit detection with a two-register stage.**

```verilog
module uart_receiver_sync #(parameter CLK_FREQ = 50_000_000, parameter BIT_RATE = 115_200) (
  input wire clk,
  input wire rx,
  output reg start_detected
);

  reg [1:0] rx_sync;

  always @(posedge clk) begin
    rx_sync <= {rx_sync[0], rx};
  end

  always @(posedge clk) begin
    start_detected <= (rx_sync[1] == 1'b1) && (rx_sync[0] == 1'b0);
  end

endmodule
```

Here, the `rx_sync` register delays the incoming `rx` signal by two clock cycles. The `start_detected` signal asserts only when a falling edge is detected on the registered input. This simple detection logic contributes to at least two clock cycles of delay before a start bit is even registered. The time it takes to detect the start bit becomes significant, especially if the simulation time resolution is not substantially finer than the system clock.

**Example 2: Oversampling logic**

```verilog
module uart_oversampler #(parameter CLK_FREQ = 50_000_000, parameter BIT_RATE = 115_200) (
  input wire clk,
  input wire data_in,
  input wire start_bit_detected,
  output reg sample_data
);

  localparam OVERSAMPLE_RATE = CLK_FREQ / BIT_RATE;
  reg [31:0] counter;
  reg sample_enable;

  always @(posedge clk) begin
    if (start_bit_detected) begin
        counter <= 0;
        sample_enable <= 1'b1;
    end else if (sample_enable) begin
        if (counter == (OVERSAMPLE_RATE / 2)) begin
            sample_data <= data_in; //sample at mid bit
            counter <= counter + 1;
        end else if (counter == (OVERSAMPLE_RATE - 1)) begin
            sample_enable <= 1'b0;
            counter <= 0;
        end
        else begin
            counter <= counter + 1;
        end
    end
  end
endmodule
```

This oversampling logic attempts to sample the incoming data at roughly the center of each bit time by counting clock cycles. It introduces further delay to allow the counter to reach the midpoint of the bit-time sampling window, effectively adding to convergence time. This delay is necessary to avoid sampling during the transition edge of each bit. The overall latency of this process is directly dependent on the oversampling factor, which influences convergence speed.

**Example 3: Complete Receiver Logic (simplified)**

```verilog
module uart_receiver #(parameter CLK_FREQ = 50_000_000, parameter BIT_RATE = 115_200) (
  input wire clk,
  input wire rx,
  output reg [7:0] received_data,
  output reg data_valid
);

  reg start_detected;
  reg [7:0] data_reg;
  reg [2:0] bit_counter;

  uart_receiver_sync sync_unit (.clk(clk), .rx(rx), .start_detected(start_detected));
  uart_oversampler oversampler_unit (.clk(clk), .data_in(rx), .start_bit_detected(start_detected), .sample_data(rx_sampled));

  always @(posedge clk) begin
    if(start_detected) begin
       bit_counter <= 0;
       data_valid <= 1'b0;
       data_reg <= 0;
    end else if(start_detected && (bit_counter < 8)) begin
        data_reg <= {data_reg[6:0], rx_sampled};
        bit_counter <= bit_counter + 1;
    end else if (bit_counter == 8) begin
         received_data <= data_reg;
         data_valid <= 1'b1;
    end
  end
endmodule
```
This top-level module utilizes the previous two. The overall latency before a complete byte is available at the output adds up through the synchronization logic, oversampling logic, and data storage. The accumulation of latency at each step contributes to a long perceived convergence time, particularly evident when analyzing in a digital simulator.

To optimize and understand Verilog UART behavior, I recommend consulting resources detailing digital clock domain crossing (CDC) methodologies. These resources address synchronization issues when transferring asynchronous signals into a synchronous environment. Exploring different oversampling techniques with adjustable sampling rates can also significantly influence the convergence times and error tolerance. I also suggest studying timing diagram techniques in detail and practicing detailed simulation with varying clock and baud rates. Finally, careful consideration of state machine design within the receiver can minimize unnecessary delays during bit recovery. Understanding these aspects will contribute towards improved UART implementations.
