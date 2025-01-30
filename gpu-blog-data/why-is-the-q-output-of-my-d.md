---
title: "Why is the Q output of my D flip-flop in Structural Verilog 'Z'?"
date: "2025-01-30"
id: "why-is-the-q-output-of-my-d"
---
The root cause of a ‘Z’ output from a D flip-flop instantiated in Structural Verilog, despite what may appear to be correct connections, often lies in incomplete or inappropriate initialization of the flip-flop’s internal storage element. This differs significantly from behavioral modeling, where initial values are often implied or easily set. Structural Verilog focuses on the *connections* between instantiated components, leaving the specific behavior of those components to their individual definitions.

I've encountered this issue numerous times when migrating behavioral models to synthesizable RTL. My first experience was with a complex memory controller, where I spent hours debugging what I thought were incorrect wiring configurations, only to find that the flip-flops were simply not being reset correctly at start-up, leading to indeterminate 'Z' states.

In digital circuits, a D flip-flop's Q output reflects the stored value at its internal node. This internal node, typically built from transistors arranged as a latch, starts in an undefined state after power-up or reset. If no reset logic is explicitly used within the flip-flop instantiation, and no default value is specified at design entry, the flip-flop’s output will remain in this high-impedance ‘Z’ state indefinitely, assuming the D input transitions from any state after power on, until it is explicitly forced to a deterministic logic ‘0’ or ‘1’ value through proper reset implementation. The ‘Z’ state signifies that the output is not actively driving either a logical high or a logical low, effectively disconnecting it from the rest of the circuit, from the digital logic perspective.

Structural Verilog builds circuits from interconnected modules, not behavior. Consider the standard instantiation of a D flip-flop. We're primarily concerned with interconnecting the D input, clock input, and Q output to other logic elements. We're *not* directly modeling the internal transistor logic itself, or any initialisation behavior; those are part of the pre-defined module of flip-flop. Here's an example using an assumed `dff` module, a flip-flop with no explicit reset:

```verilog
module top_level(input d, input clk, output q);
  wire q_int;
  dff my_flipflop (
    .d(d),
    .clk(clk),
    .q(q_int)
  );

  assign q = q_int;
endmodule
```

In this example, `my_flipflop` is an instance of `dff` (which can come from a vendor IP or a library). Critically, the flip-flop itself doesn't include any reset logic. The initial state of `q_int`, and hence `q`, will be 'Z' because there is no reset or default defined *within the `dff` module*.  The transition of `d` from low to high or high to low alone will not initialize `q`, it will latch any logic value once the clock edge arrives.

To resolve this, we generally incorporate an *asynchronous* or *synchronous* reset input to the D flip-flop. Let’s implement an example with an asynchronous reset:

```verilog
module top_level(input d, input clk, input rst_n, output q);
  wire q_int;
  dff_async_reset my_flipflop (
    .d(d),
    .clk(clk),
    .rst_n(rst_n),
    .q(q_int)
  );

  assign q = q_int;
endmodule

module dff_async_reset(input d, input clk, input rst_n, output q);
  reg q_reg;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q_reg <= 1'b0; // reset low on active low reset.
    end
    else begin
        q_reg <= d;
    end
  end
  assign q = q_reg;
endmodule

```

Here, `dff_async_reset` explicitly includes a reset input `rst_n`. When `rst_n` is low, the flip-flop will be forced to a logical ‘0’, overriding the normal data-capture action, and preventing the default ‘Z’ state. The reset signal must be activated at circuit initialization, and held until a stable power state is reached, before the device can move out of reset.

It’s worth noting that a *synchronous* reset can also be used, which has different timing and implementation concerns. The difference is that in a synchronous reset, the reset event must also be synchronized with the clock edge:

```verilog
module top_level(input d, input clk, input rst, output q);
  wire q_int;
  dff_sync_reset my_flipflop (
    .d(d),
    .clk(clk),
    .rst(rst),
    .q(q_int)
  );

  assign q = q_int;
endmodule

module dff_sync_reset(input d, input clk, input rst, output q);
  reg q_reg;
  always @(posedge clk) begin
    if (rst) begin
      q_reg <= 1'b0; // reset low on active high reset.
    end
    else begin
        q_reg <= d;
    end
  end
  assign q = q_reg;
endmodule
```
In this `dff_sync_reset` model, the `q` output is only reset if `rst` is high at the *positive edge* of `clk`. This makes for a more robust design, but with its own challenges in timing and routing. In this example the reset is active-high rather than active-low as in the previous example.

The important takeaway is that a 'Z' output in Structural Verilog is a sign that a flip-flop, or more specifically its stored data, has not been properly initialized.  This is often due to the lack of reset logic and requires an intentional design choice, either the use of a module that instantiates a flip-flop with a default value, or via the inclusion of an explicit reset signal within the logic.

For further exploration of this topic, I recommend consulting the Verilog standard (IEEE 1364) documentation, particularly the sections detailing flip-flop behavior, reset strategies, and the difference between behavioral and structural modeling. Also, refer to texts covering digital circuit design, these often delve deeper into the nuances of flip-flop internal operation. Synthesizable RTL guidelines are also indispensable, detailing correct methods for specifying reset circuitry and module initialization to ensure proper hardware behavior after synthesis. Examining the synthesis reports often details the actual physical implementations of design elements and is invaluable in debugging these kinds of issues. Finally, the datasheets of your target FPGA or ASIC provide explicit details on supported flip-flop modules and recommended initialization processes.
