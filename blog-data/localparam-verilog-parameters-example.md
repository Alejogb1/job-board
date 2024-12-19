---
title: "localparam verilog parameters example?"
date: "2024-12-13"
id: "localparam-verilog-parameters-example"
---

Alright so localparam in Verilog right I've been there wrestled with that beast more times than I care to admit

Yeah its a common headache for beginners and sometimes for seasoned pros too when you're knee-deep in a complex design and forgetting your basic syntax localparams are those nifty little constants you define within a module or a package they're like the silent workhorses of parameterized hardware design allowing you to configure things without the hassle of re-writing massive chunks of code each time

Let's say you're building an adder a really basic one just to make it clear You could do it the old school way hardcoding the bit width everywhere but that's a maintenance nightmare waiting to happen what if you need a wider adder later on you'd have to go through every single line and change the bit width everywhere. That's no good at all.

```verilog
module bad_adder (input  [7:0] a,
                 input  [7:0] b,
                 output [7:0] sum);

  assign sum = a + b;
endmodule
```

See that 7:0 that's what we call a magic number it works here but it's also really bad coding practice using magic numbers. No bueno. Now consider this scenario

```verilog
module better_adder #(parameter WIDTH = 8)
                (input  [WIDTH-1:0] a,
                 input  [WIDTH-1:0] b,
                 output [WIDTH-1:0] sum);

  assign sum = a + b;
endmodule
```

This is much better we have a parameter called WIDTH and we can change it when we instantiate the module but what if I don't want to change the width outside the module only within the module I have different internal uses for the width well here is when we have the local parameters

Let's look at some localparam examples and you'll quickly get the idea I remember back in my early days working on some FPGA projects I kept mixing up parameter declarations with localparams and it led to all sorts of crazy errors hours of debugging wasted I remember one instance in particular where I wanted to define a specific address range in my memory mapped module using parameters I defined it as a regular parameter I instantiated my module in the top level and then tried to change the address within the module by assigning to it via a generate statement I ended up generating an error of not being able to change a parameter. Turns out I needed to use localparam to not allow this to happen from the top level.

Here's a simple example say we're building some sort of communication interface with a specific address size and a data width and we want some local calculations

```verilog
module comm_interface (
  input  [31:0]  data_in,
  output [31:0]  data_out,
  input        clk,
  input        reset
);

localparam ADDR_WIDTH = 16;
localparam DATA_WIDTH = 32;
localparam FULL_ADDR = (1 << ADDR_WIDTH) - 1;

reg [DATA_WIDTH - 1:0] data_reg;

always @(posedge clk) begin
  if (reset) begin
    data_reg <= 0;
  end else begin
    data_reg <= data_in;
  end
end

assign data_out = data_reg;

endmodule
```
See here ADDR\_WIDTH and DATA\_WIDTH are localparams and FULL\_ADDR is a local param that uses the other localparams you can only define parameters that way within the localparam definition not as a variable so you need to be careful here it is something you cant change from outside the module and this is a good thing as you should not change it there. The FULL\_ADDR calculation is done at compile time so you don't get a runtime performance hit.

Let's make it a tad more complex adding another module that we are going to generate

```verilog
module complex_interface #(parameter INTERFACE_WIDTH = 8) (
  input   [INTERFACE_WIDTH-1:0] data_in,
  output  [INTERFACE_WIDTH-1:0] data_out,
  input        clk,
  input        reset
);

  localparam DATA_WIDTH = INTERFACE_WIDTH * 2;
  localparam SUB_WIDTH = INTERFACE_WIDTH / 2;
  localparam NUM_SUB_MODULES = DATA_WIDTH / SUB_WIDTH;

  reg [DATA_WIDTH-1:0] internal_data;

  generate
    for(genvar i = 0; i < NUM_SUB_MODULES; i = i + 1) begin : submodules
      sub_module #(
      .SUB_WIDTH(SUB_WIDTH)
      )
      sub_inst (
        .data_in(internal_data[i*SUB_WIDTH +: SUB_WIDTH]),
        .data_out(data_out[i*SUB_WIDTH +: SUB_WIDTH]),
        .clk(clk),
        .reset(reset)
      );
    end
  endgenerate

  always @(posedge clk) begin
      if (reset) begin
        internal_data <= 0;
      end else begin
        internal_data <= {internal_data[DATA_WIDTH-SUB_WIDTH-1:0],data_in};
      end
   end
endmodule

module sub_module #(parameter SUB_WIDTH = 4) (
  input [SUB_WIDTH-1:0] data_in,
  output [SUB_WIDTH-1:0] data_out,
  input clk,
  input reset
);

  reg [SUB_WIDTH-1:0] sub_reg;

  always @(posedge clk) begin
      if (reset) begin
          sub_reg <= 0;
      end else begin
        sub_reg <= data_in;
      end
  end

  assign data_out = sub_reg;

endmodule
```
Here `DATA_WIDTH` `SUB_WIDTH` `NUM_SUB_MODULES` are all localparams we are using local params to determine the amount of sub modules we are instantiating and the widths of the signals inside those submodules. The `genvar i` is another parameter but that is not what we are talking about here. If this were a top-level parameter you would not be able to change the amount of sub modules instantiated. I once spent half of my day debugging this exact thing. Turns out parameters are for top level instantiation adjustments not local adjustments within modules.

Key takeaways are that `localparam` are constants within a module or a package they are not meant to be changed by higher levels of design hierarchy and are meant to be used for internal calculations that should not be changed by users or other modules and help in making code more maintainable by assigning specific names to constants and it is also important to note that this is all calculated during compile time. And parameters are meant to adjust the module behavior from top level.

Another thing I remember I did was to set the value of a local parameter based on another parameter I am not sure if this is a good idea or not but its fun and can be handy.
```verilog
module parameterized_localparam #(parameter BASE_VALUE = 10) (
  input  [31:0]  data_in,
  output [31:0]  data_out,
  input        clk,
  input        reset
);

  localparam OFFSET_VALUE = BASE_VALUE * 2;
  localparam RESULT_VALUE = BASE_VALUE + OFFSET_VALUE;

  reg [31:0] internal_reg;

  always @(posedge clk) begin
    if (reset) begin
      internal_reg <= 0;
    end else begin
      internal_reg <= data_in + RESULT_VALUE;
    end
  end

  assign data_out = internal_reg;

endmodule
```
here `OFFSET_VALUE` and `RESULT_VALUE` are derived from the `BASE_VALUE` parameter this example is pretty dumb but you get the idea it can be really useful in other real world cases.

Regarding resources for Verilog specifically look for books like "Digital Design and Computer Architecture" by David Money Harris and Sarah L Harris it goes a good length on hardware descriptions with verilog and "SystemVerilog for Verification" by Chris Spear. There are also many online tutorials but i found that for this level of hardware design books and well structured papers are better. Also don't forget to always be looking at the standard documents there is something in there that always help even the most advanced engineers. It is the standard for a reason I guess. One day a senior engineer joked with me that the only thing he ever reads is the standard documents and he is being completely serious. I thought that was funny.

I hope that helped clear things up and good luck with your hardware design journeys.
