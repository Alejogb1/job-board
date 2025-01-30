---
title: "Why is the Artix-7 FPGA missing in Vivado for the Arty A7-100T?"
date: "2025-01-30"
id: "why-is-the-artix-7-fpga-missing-in-vivado"
---
The Arty A7-100T development board, while featuring an Artix-7 FPGA, doesn't explicitly list 'Artix-7' as a selectable part in the Vivado device selection dropdown. This is a common point of confusion for new users familiar with the 'Artix-7' name but not the specific naming convention used by Xilinx for their devices. The discrepancy stems from the fact that Vivado, Xilinx's design suite, identifies FPGAs by their precise part numbers, not broad architectural families. The Arty A7-100T specifically uses the XC7A100T-1CSG324C, and this exact part number must be selected to properly target the board.

When beginning a new Vivado project, the initial dialog asks you to choose a specific target device. Rather than offering a high-level option like "Artix-7," Vivado presents a list of individual part numbers. Each number represents a specific configuration of the silicon, including the amount of logic resources, available memory blocks, I/O pins, and performance grades. Confusing family names with exact part numbers is akin to referencing an "Intel Core i7" instead of the precise "Intel Core i7-13700K." They both belong to the same family, but the specifics are critical. Therefore, the root issue isn't the omission of Artix-7; it's the necessary selection of the specific part number, which is then correctly associated with the Artix-7 family within the Vivado ecosystem.

Furthermore, device-specific board files supplied by Digilent, the Arty board’s manufacturer, rely on this specific part number. These board files contain critical information such as pin assignments, clock locations, and memory interfaces. If you were to incorrectly select a part number, even another variant within the Artix-7 family, many of these constraints would be mismatched, leading to a non-functional design even if the code itself appeared correct. Think of it as a template for the physical hardware, matching each line of code to an exact pin on the chip. Without the correct template, the system won't "understand" where the code should be implemented.

To illustrate, let's look at creating a simple Verilog project for the Arty A7-100T. This example shows how the correct part selection directly affects the project setup.

**Example 1: Incorrect Device Selection (Hypothetical)**

```verilog
module led_blink (
    input  wire clk,
    output reg led
);

  reg [27:0] counter;

  always @(posedge clk) begin
    counter <= counter + 1;
    if (counter == 28'd50000000) begin
        counter <= 28'd0;
        led <= ~led;
    end
   end

endmodule
```

This Verilog code describes a simple LED blinking circuit. If you were to mistakenly select, for example, an XC7A50T part number within the Artix-7 range, rather than the correct XC7A100T-1CSG324C, the following would happen during synthesis or implementation. Vivado would use the resources available for an XC7A50T, which has a smaller footprint, potentially causing routing issues due to constraints mismatches. The pin assignment would be incorrect, and your LEDs won’t function. This code itself is correct, but the hardware context within which it's being placed is wrong. You’d likely see warnings or errors indicating pin conflicts or that the target device doesn’t have the pins being constrained, which provides a clue of a deeper part number mismatch.

**Example 2: Correct Device Selection and Constraint File (XDC)**

To ensure correct implementation, the project must be configured to use XC7A100T-1CSG324C and an appropriate constraint file, also known as an XDC file, is required. Here's an example of the relevant constraint section:

```xdc
# LED 0
set_property -dict { PACKAGE_PIN H17   IOSTANDARD LVCMOS33 } [get_ports { led }];
```
This XDC section explicitly connects the `led` output in our Verilog code to pin H17 on the FPGA, a pin that is physically connected to a user LED on the Arty A7-100T board. This assumes the correct device was selected.

**Example 3: Correct Device Selection and Project Configuration**

Here is a breakdown of the crucial steps:

1.  Create a New Vivado Project.
2.  In the 'Project Type' step, select 'RTL Project'.
3.  In the 'Default Part' step, the crucial step:
    * Instead of looking for 'Artix-7,' expand the 'Parts' section.
    * Choose 'Boards' for an even simpler project setup.
    * Select 'Arty A7-100T'. This step automates part selection.
    * Alternatively, you can search directly by part number if you choose "Parts" over "Boards": XC7A100T-1CSG324C.
4.  Add the Verilog code in Example 1 to your project (ensure a top module is set if you have multiple modules).
5.  Add the XDC constraint file in Example 2, to ensure the correct pin assignments and IO standard for the LED output.
6.  Generate the bitstream and load it on the Arty A7 board.

The act of selecting the right board or exact part number ensures Vivado is aware of the specific physical attributes of the FPGA. This correct board selection, in turn, loads all the required constraint files in the backend, simplifying development. By targeting the XC7A100T-1CSG324C, the design will be synthesized and implemented correctly, and our led blink example will function as expected.

Incorrect device selection can lead to seemingly inexplicable errors, making it a common stumbling block. This point cannot be overstressed; double checking is a crucial aspect of FPGA development. If the correct board is selected in Vivado, you will rarely have to manually select the part number and the relevant constraint files and settings will be automatically loaded, thus minimizing confusion.

For further learning, I recommend focusing on resources detailing the Xilinx FPGA architecture and naming conventions. Specific technical documentation like the 7 Series FPGA data sheet (DS180) provides a detailed breakdown of every part number and the associated resources. Also, reading Digilent's documentation regarding the Arty A7-100T boards is key to understand its pinout and functionality. Look for manuals and user guides which provide in-depth explanations of the hardware components, board constraints, and sample project setups. Additionally, explore Vivado’s built-in help documentation to fully understand how part selection impacts the design process and how the constraints editor works. These resources should give you the necessary tools to navigate the differences between high-level family names and specific part numbers, and ultimately provide a solid foundation for your future FPGA projects. Understanding this nuanced distinction between family names and exact part numbers is critical for successful FPGA development.
