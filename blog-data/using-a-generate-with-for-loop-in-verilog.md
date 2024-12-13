---
title: "using a generate with for loop in verilog?"
date: "2024-12-13"
id: "using-a-generate-with-for-loop-in-verilog"
---

Okay so you're asking about using `generate` with `for` loops in Verilog yeah I get it I've been there done that got the t-shirt a few times I mean I've probably coded more hours in Verilog than I've slept in the last decade maybe more than that I'm kidding okay maybe not anyway let's get to it because this is a really common pattern when you're dealing with hardware descriptions

See the thing about hardware design with HDL is it's not just about getting it to *work* it's about making it work *efficiently* and *scalably* you know you don't want to be manually writing out a hundred instantiations of the same module when you can just loop it right? That's where `generate` with `for` comes in it's a powerful construct for generating repetitive structures in your Verilog code and for creating parameterized designs.

I mean remember that time I was working on this custom ASIC for an AI accelerator? We needed a huge array of processing elements and believe me manually wiring all those up that’s a special type of pain right? We even had that intern who tried to copy paste the module a couple of times before we intervened haha Anyway using generate with for loop was our salvation it saved us weeks of work and probably a couple of more gray hairs.

Alright so basically the `generate` block lets you create multiple instances of modules or logic based on a condition or loop variable while the `for` loop lets you iterate through a range of values. When you combine the two you can automatically create a bunch of identical or slightly modified blocks in your design.

Let’s start with a super basic example because everyone loves a good example right? Imagine you want to create a bunch of identical flip-flops which is a common thing to do.

```verilog
module flip_flop (
  input clk,
  input d,
  output reg q
);
  always @(posedge clk) begin
    q <= d;
  end
endmodule

module multiple_flops (
  input clk,
  input [7:0] d,
  output [7:0] q
);
  genvar i;
  generate
    for (i = 0; i < 8; i = i + 1) begin : flip_flop_gen
      flip_flop ff (
        .clk(clk),
        .d(d[i]),
        .q(q[i])
      );
    end
  endgenerate
endmodule

```

In this code I define a basic `flip_flop` module and then in the `multiple_flops` module I used the `generate` `for` loop with a `genvar` variable to instantiate eight flip-flops connecting their data inputs to different bits of the input data vector d and the output to the output vector q This `genvar` i is the loop variable and it’s not a real signal in the hardware it’s just used at compile time to repeat the instantiation block. Notice the line where I use `begin : flip_flop_gen`? This is important because it gives a name `flip_flop_gen` to each instance of the generated block allowing you to debug more easily when dealing with large designs.
Now let’s get a bit more complex because that example is very basic. How about a parameterized adder right?

```verilog
module adder_module #(parameter WIDTH = 8) (
    input [WIDTH-1:0] a,
    input [WIDTH-1:0] b,
    output [WIDTH-1:0] sum
);
  assign sum = a + b;
endmodule


module multiple_adders #(parameter NUM_ADDERS = 4, parameter WIDTH = 8)(
  input clk,
  input [NUM_ADDERS*WIDTH-1:0] a,
  input [NUM_ADDERS*WIDTH-1:0] b,
  output [NUM_ADDERS*WIDTH-1:0] sum
);
  genvar i;
  generate
    for (i=0; i<NUM_ADDERS; i=i+1) begin : adders_gen
      adder_module #(
          .WIDTH(WIDTH)
      ) adder (
          .a(a[i*WIDTH +: WIDTH]),
          .b(b[i*WIDTH +: WIDTH]),
          .sum(sum[i*WIDTH +: WIDTH])
      );
    end
  endgenerate

endmodule
```
Here you see the `adder_module` that takes a `WIDTH` parameter and adds two numbers of that `WIDTH` then in the `multiple_adders` module we take the `NUM_ADDERS` and the `WIDTH` parameters we also have an input `a`, `b` and `sum` to connect the different instances of adders created by the generate for loop We used  the `+:` bit select to address the different bits of the input/output bus. You may notice that I could parameterize the adder module instead of using the bit select. That would certainly work but in this example I wanted to demonstrate how to use bit selects inside a generate for loop.

Okay one more a bit more complicated example because why not lets do a shift register with a variable length using `generate` and `for`. This time with enable logic because that is always needed.

```verilog
module shift_register #(parameter LENGTH = 8)(
    input clk,
    input enable,
    input data_in,
    output data_out
);
  reg [LENGTH-1:0] shift_reg;
  always @(posedge clk) begin
    if(enable) begin
      shift_reg <= {shift_reg[LENGTH-2:0], data_in};
    end
  end
  assign data_out = shift_reg[LENGTH-1];
endmodule

module multiple_shift_registers #(parameter NUM_REGISTERS = 4, parameter LENGTH = 8)(
  input clk,
  input enable,
  input [NUM_REGISTERS-1:0] data_in,
  output [NUM_REGISTERS-1:0] data_out
);
  genvar i;
  generate
    for (i=0; i<NUM_REGISTERS; i=i+1) begin : shift_reg_gen
      shift_register #(
          .LENGTH(LENGTH)
      ) shift_reg (
          .clk(clk),
          .enable(enable),
          .data_in(data_in[i]),
          .data_out(data_out[i])
      );
    end
  endgenerate
endmodule

```
This time we have a `shift_register` module that takes the parameter `LENGTH` and shifts the input `data_in` to the left each clock cycle if the enable signal is high and then in the `multiple_shift_registers` module using `generate` and `for` loops we instantiate N shift registers that take the input from the data_in bus and output to data_out bus.

Now a few gotchas I've encountered over time. First be careful with your loop condition if you make an infinite loop inside generate the synthesis tool will go into an infinite loop which is no bueno trust me I once waited for three hours before killing the process. Second always remember that the loop variables inside the generate block are evaluated at compile time not run-time you cannot use a value coming from your design. The generate for block is essentially a kind of code macro. Lastly try to give proper names to your generate blocks it will save you tons of headaches.

And speaking of gotchas do not get your wires crossed when using generate blocks with module instantiation you might end up connecting the wrong things. It is a classic error that might end up causing hours of debugging. The most common is to connect to the wrong bit of a bus.

For further information I suggest the following resources:

*   **"SystemVerilog for Verification" by Chris Spear:** This book while focused on verification has some very nice explanations and examples of code generation and parametrization.
*   **"Digital Design Principles and Practices" by John F Wakerly:** This is a good introductory book but it has a nice explanation on how these hardware description languages works and how they work at a higher level
*   **IEEE Std 1800-2017:** This is the official language reference standard and has the most comprehensive explanation but you have to be used to reading standards

Okay that should keep you busy for a while. Just remember to think of the hardware when writing Verilog and you will be golden.

Hope that helps let me know if you have any other questions I've seen it all.
