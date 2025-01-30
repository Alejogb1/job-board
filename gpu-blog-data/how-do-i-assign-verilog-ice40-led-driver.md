---
title: "How do I assign Verilog ICE40 LED driver signals to SB_IO_OD?"
date: "2025-01-30"
id: "how-do-i-assign-verilog-ice40-led-driver"
---
The ICE40 FPGA architecture requires careful handling of its I/O primitives, particularly when driving LEDs due to their current requirements. Direct assignment of LED driving signals to the standard SB_IO input/output block is generally unsuitable because it lacks the necessary current sinking capability. Instead, the SB_IO_OD primitive, a dedicated open-drain output buffer, should be employed. This primitive necessitates external pull-up resistors to properly drive the LED and its current. Improper implementation can lead to unpredictable behavior or damage to the FPGA and connected components. I've encountered this frequently when designing embedded controllers and noticed common pitfalls in initial attempts to drive LEDs.

The primary reason for using SB_IO_OD is its specific electrical characteristic: an open-drain configuration. Unlike standard CMOS outputs, which actively drive both high and low states, an open-drain output only drives low, effectively acting as a switch to ground. This means the high state must be passively established through a pull-up resistor. The benefit here is that we control the current flow through the LED with this external resistor, preventing damage to the FPGA and the LED itself. Furthermore, the current-sinking capability of a standard SB_IO output is typically insufficient to directly drive an LED. Attempting this can result in the output struggling, potentially leading to signal integrity issues or even causing the output stage to overheat, reducing the lifespan of the FPGA.

To utilize the SB_IO_OD primitive, you'll first declare it in your Verilog code, specifying necessary attributes. Then, you’ll connect your intended LED driver signal to the 'O' output of the primitive. This signal will then be the open-drain output, which requires a pull-up resistor connected between the LED's anode and your supply voltage. The LED's cathode will be connected to the FPGA pin assigned to the SB_IO_OD primitive. It is vital to correctly compute the resistor value depending on the desired LED current. This is calculated by Ohm’s law, considering the LED's forward voltage drop and the supply voltage used. This calculation is specific to each use case.

Here are three Verilog examples illustrating different facets of this process:

**Example 1: Basic LED Drive with SB_IO_OD**

```verilog
module led_driver_basic (
    input clk,
    input reset,
    output led_out
);

  reg led_state;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      led_state <= 1'b0;
    end else begin
       led_state <= ~led_state;
    end
  end

  SB_IO_OD #(
    .PIN_TYPE ( 6'b000001 ),
    .PULLUP   ( 1'b0 )
  ) led_io (
    .O  ( led_state ),
    .I  ( 1'b0 ) // Input is not used for output only functionality
  );
  
  assign led_out = led_io.O; // Connect the output of the SB_IO_OD to the physical output pin
endmodule
```

In this basic example, a register, `led_state`, toggles its value on every clock cycle. This register's value is connected to the 'O' output of the SB_IO_OD primitive. Note the `PIN_TYPE` parameter is set to `6'b000001`, indicating a dedicated output, and the `PULLUP` parameter is `1'b0`, because we will add an external one.  The 'I' input of the `SB_IO_OD` is tied to ground, because we do not use the input functionality, this is typical when you use it for output only. The `led_out` signal, which will go to a pin, is the output from the SB_IO_OD module. On the hardware side, an LED and a pull-up resistor (value based on desired current) will be connected between the output pin and your power supply. The logic high from the pull up resistor completes the circuit and drives the LED when `led_state` is low.

**Example 2:  Using `SB_GB_IO` for Global Clock Input and Driving Multiple LEDs**

```verilog
module led_driver_multi (
  input clk_in,
  input reset,
  output [2:0] led_out
);

  wire clk;

  SB_GB_IO #( .PIN_TYPE (6'b100100) ) clk_buf (.I(clk_in), .O(clk));
  
  reg [2:0] led_state;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      led_state <= 3'b000;
    end else begin
       led_state <= led_state + 3'b001;
    end
  end
  
  SB_IO_OD #(
    .PIN_TYPE ( 6'b000001 ),
    .PULLUP   ( 1'b0 )
    ) led_io0 ( .O  ( led_state[0] ), .I  ( 1'b0 ) );
    
   SB_IO_OD #(
    .PIN_TYPE ( 6'b000001 ),
    .PULLUP   ( 1'b0 )
    ) led_io1 ( .O  ( led_state[1] ), .I  ( 1'b0 ) );
    
    SB_IO_OD #(
    .PIN_TYPE ( 6'b000001 ),
    .PULLUP   ( 1'b0 )
    ) led_io2 ( .O  ( led_state[2] ), .I  ( 1'b0 ) );

  assign led_out[0] = led_io0.O;
  assign led_out[1] = led_io1.O;
  assign led_out[2] = led_io2.O;

endmodule
```
This example expands on the previous one. It demonstrates using `SB_GB_IO`, a global buffer, to prepare the incoming clock signal using the `PIN_TYPE` parameter `6'b100100`, for global clock input. It then drives three LEDs controlled by `led_state` using three `SB_IO_OD` instances.  The counter within `led_state` increments, creating a more complex visual pattern on the LEDs and demonstrating multiple `SB_IO_OD` primitives instantiation within the same module. Note how the output of each `SB_IO_OD` is assigned to the vector output, which would be connected to three physical pins. Again, external resistors and LEDs are required.

**Example 3:  Parameterized LED Driver Module**

```verilog
module parameterized_led_driver #(
    parameter LED_COUNT = 4
) (
  input clk,
  input reset,
  output [LED_COUNT-1:0] led_out
);

  reg [LED_COUNT-1:0] led_state;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      led_state <= {LED_COUNT{1'b0}};
    end else begin
      led_state <= led_state + 1;
    end
  end

  genvar i;
  generate
    for (i = 0; i < LED_COUNT; i = i + 1) begin: led_gen
        SB_IO_OD #(
        .PIN_TYPE ( 6'b000001 ),
        .PULLUP   ( 1'b0 )
        ) led_io ( .O  ( led_state[i] ), .I  ( 1'b0 ) );

         assign led_out[i] = led_io.O;
    end
  endgenerate

endmodule
```

This example introduces parameterization and `generate` loops. The number of LEDs driven is now defined by the `LED_COUNT` parameter. The `generate` block dynamically creates the required number of `SB_IO_OD` instances and output assignments based on this parameter, demonstrating an approach to create scalable and reusable LED driving modules. When instantiating this module, the parameter can be changed during instatiation, for example in a testbench, allowing driving of a variable number of LEDs with the same module.  As before, pull-up resistors and LEDs are needed on the board. The instantiation of this module would require a parameter assignment during module instantiation, for example `parameterized_led_driver #( .LED_COUNT(8) ) my_led_driver ( /*port connections*/ );` to create 8 LED drivers.

These examples highlight the important aspects of using `SB_IO_OD` to drive LEDs. These examples should provide a firm base to work from when designing your own modules. In conclusion, careful consideration of the electrical characteristics and proper utilization of the `SB_IO_OD` primitive are paramount for reliable LED driving with ICE40 FPGAs.

For additional information, refer to the Lattice ICE40 FPGA device documentation, available from Lattice Semiconductor's website. Specifically, consult the device handbook, which details the various primitives and their parameters. Additionally, search for application notes or reference designs from Lattice, which may present real-world use cases and detailed example configurations and constraints, though these may not all explicitly cover LEDs and open-drain functionality. Also, numerous online FPGA communities or websites can provide assistance with specific problems that might arise when using these devices, though be sure to critically evaluate the information found online.

Finally, ensure you understand the electrical characteristics of the LEDs you are using and the resulting requirements on the external resistor.
