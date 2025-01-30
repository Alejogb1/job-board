---
title: "How can a Verilog code replicate an input square wave?"
date: "2025-01-30"
id: "how-can-a-verilog-code-replicate-an-input"
---
A critical design consideration when implementing digital systems involves generating reliable clock signals. Directly replicating an input square wave in Verilog, although seemingly straightforward, requires careful handling of edge cases and potential metastability issues. I’ve encountered this exact problem when designing a high-speed data acquisition system, where a clean reference clock derived from an external source was necessary for synchronization.

The fundamental challenge lies not in simply mirroring the input; rather, it’s about ensuring that the replicated signal maintains the intended frequency and duty cycle while avoiding glitches and other undesirable artifacts that can arise due to asynchronous transitions and finite propagation delays within the digital logic. A naïve approach of directly assigning the input to the output often results in an unusable clock due to the inherent delays and potential for race conditions in the synthesized hardware. Consequently, a more robust solution is required.

The core technique I've found most effective involves using a flip-flop to synchronize the input signal. This effectively re-clocks the incoming square wave using the internal clock domain of the FPGA or ASIC. This process mitigates the risk of metastability, where the output of a flip-flop can remain in an indeterminate state for a period of time when its data and clock inputs transition close together. This can lead to unpredictable behavior. To prevent this, we use a technique called "synchronization," where the input is sampled multiple times with the internal clock to ensure a stable value. This is not a method to change the input frequency, rather to align it to the local clock domain.

The simplest and most common implementation involves a two-stage synchronizer, using two flip-flops in series. While more stages can further reduce metastability probability, two stages usually provide a good balance of robustness and resource utilization. The first flip-flop samples the input using the rising edge of the internal clock, and the second captures the output of the first, also on the rising edge of the same internal clock. The output of the second flip-flop is then used as the synchronized version of the input square wave.

The first code example presents a basic two-stage synchronizer. This is applicable for the case where you just require a replica clock and are not concerned with signal frequency or phase:

```verilog
module square_wave_replicator_basic(
    input  wire  input_clk,
    input  wire  internal_clk,
    output reg  output_clk
);

reg sync_reg1;
reg sync_reg2;

always @(posedge internal_clk) begin
  sync_reg1 <= input_clk;
  sync_reg2 <= sync_reg1;
end

assign output_clk = sync_reg2;

endmodule
```
In this example, `input_clk` represents the external square wave to be replicated, and `internal_clk` is the internal clock signal of the target hardware. `sync_reg1` and `sync_reg2` are intermediate registers used for synchronization. The `always` block samples the `input_clk` signal using the rising edge of `internal_clk` and propagates the sampled value through the two registers. The output `output_clk` is the synchronized replica of `input_clk`, aligned to the rising edge of `internal_clk`. Note that the frequency of the `output_clk` remains that of `input_clk`, though phase and any jitter from the input will be absorbed by the internal clock.

The next code example demonstrates how to generate a square wave of specified frequency and duty cycle. Note that this is *not* replicating an external clock. Instead it provides an example of creating a clock signal using counters, which can be useful in internal parts of a system. This is a simple example that does not incorporate sophisticated methods of precise duty cycle adjustment and frequency synthesis:

```verilog
module square_wave_generator(
    input   wire  internal_clk,
    input   wire [31:0] freq_divider,
    input   wire [31:0] duty_divider,
    output  reg         output_clk
);

reg [31:0] counter;

always @(posedge internal_clk) begin
    if (counter == freq_divider - 1) begin
        counter <= 0;
        output_clk <= ~output_clk; //Toggle Output
    end else begin
        counter <= counter + 1;
        if (counter < duty_divider) output_clk <= 1'b1;
          else output_clk <= 1'b0;
    end
end

endmodule
```

In this module, `internal_clk` provides the timing reference. `freq_divider` dictates the frequency, where the period of the output clock is directly proportional to this. The `duty_divider` sets the number of cycles within the output clock cycle when the output will be HIGH. The `counter` keeps track of the clock cycles and, when it reaches `freq_divider-1`, is reset to `0` and the `output_clk` toggles. This module is not used for synchronization of an *external* square wave signal, but for generating a square wave of a defined period internally.

Finally, the third example, builds upon the first example. This example demonstrates a more robust approach, including a reset and an output enable signal, while demonstrating the usage of non-blocking assignments:

```verilog
module robust_square_wave_replicator(
    input   wire      input_clk,
    input   wire      internal_clk,
    input   wire      reset,
    input   wire      enable,
    output  reg       output_clk
);

reg sync_reg1;
reg sync_reg2;

always @(posedge internal_clk or posedge reset) begin
    if(reset) begin
        sync_reg1 <= 1'b0;
        sync_reg2 <= 1'b0;
    end else begin
      if (enable)
        begin
            sync_reg1 <= input_clk;
            sync_reg2 <= sync_reg1;
        end
      else
        begin
            sync_reg1 <= 1'b0;
            sync_reg2 <= 1'b0;
        end
    end
end

always @* begin
    output_clk = sync_reg2;
end

endmodule
```
Here, a synchronous reset, `reset`, is introduced. The reset ensures that the synchronizer's flip-flops are initialized to a known state. It is critical for preventing unknown states at the time of power-up. The input `enable` is an asynchronous signal which enables the replicator functionality. The synchronizer registers `sync_reg1` and `sync_reg2` are only updated when the `enable` signal is asserted. The second always block uses a continuous assignment to drive the output. It is important that the synchronizer flip-flops are only ever driven from an edge-triggered clock block, while the output can be combinatorially derived.

During my time implementing these, I have found several points to consider. The internal clock chosen for synchronization needs to have a frequency significantly higher than the input square wave's frequency to reduce the probability of metastability and jitter accumulation. The two-stage synchronizer method can introduce a delay of up to two internal clock cycles to the replicated clock. This delay is deterministic and can be compensated for in the downstream design if required. The specific resource utilization of these modules depends on the target technology and synthesis tools. Therefore, reviewing the synthesis reports is essential to optimize performance and resource usage.

For those seeking deeper understanding of digital design and clock management, I recommend exploring resources covering: advanced synchronization techniques, metastability handling in asynchronous circuits, and digital design principles for high-speed systems. Standard textbooks on digital logic design often detail flip-flop behavior and metastability. Vendor specific documentation relating to the target FPGA family’s clocking resources will be invaluable. Further reading on the subject of clock domain crossing (CDC) will also provide valuable context. Finally, resources focused on VHDL (Very High-Speed Integrated Circuit Hardware Description Language), a design language alternative to Verilog, can also provide unique insights.
