---
title: "difference between parameter and localparam?"
date: "2024-12-13"
id: "difference-between-parameter-and-localparam"
---

Alright let's dive into this parameter vs localparam thing I've seen this trip up a ton of folks especially when they're just getting into hardware description languages like Verilog or SystemVerilog. I've been there trust me had my fair share of head-scratching moments back in the day.

So straight up parameters and localparams they're both ways to define constants in your code but they've got some key differences in their scope and how you can modify them. Think of it like this parameter is like a configurable setting on a product it's meant to be tweaked before you actually build something. A localparam on the other hand is like an internal constant that is baked in it’s not really meant to be changed by anyone using your module.

Let's break it down.

**Parameters**

Parameters are primarily meant for making your modules reusable and adaptable. They're like module-level variables that you can modify when you instantiate your module meaning when you use that module inside other modules. You define them in your module's declaration and when you create an instance of that module you can override these parameter values. This is HUGE for creating flexible components.

For instance imagine I'm building an adder module but I want to make its input size variable I’d use a parameter here:

```verilog
module generic_adder #(parameter WIDTH = 8)
(
input  [WIDTH-1:0] a
input  [WIDTH-1:0] b
output [WIDTH-1:0] sum
);
  assign sum = a + b;
endmodule
```
In this example `WIDTH` is a parameter with a default value of 8. If I create an instance of this `generic_adder` module without specifying a value for `WIDTH` it'll default to 8. But here's the magic:

```verilog
module top_level;
  wire [7:0] sum8;
  wire [15:0] sum16;

  generic_adder adder8 ( .a(input_8a) .b(input_8b), .sum(sum8) );
  generic_adder #(.WIDTH(16)) adder16 ( .a(input_16a) .b(input_16b) .sum(sum16) );
endmodule
```

Notice how in `top_level` I have two instances of `generic_adder`. The first one `adder8` uses the default `WIDTH` of 8. The second one `adder16` explicitly changes the `WIDTH` to 16. That flexibility is what makes parameters so powerful.

I used to work on a project that involved creating configurable arithmetic logic units back in my early days. We had this huge complex unit that had to be adaptable for several microarchitectures that used it. Parameters saved us so much time we were able to create a single well defined module and then create multiple instances of it tailored to the specific needs of that part of the processor by modifying its parameters. It is a great example of the power of parameterized design.

**Localparams**

Now localparams they're different. They're still constants but they're meant to be internal to the module. You can't override them when you create an instance of your module. They are constants that are defined and used only inside that particular module and are meant to make the code more readable or help with calculation without having to declare a complex expression all over the place. This is something I've used when creating complex decoders. It's just not easy to keep track of all the bit shifting and mask combinations so using localparams will improve readability.

Here is how that looks:

```verilog
module decoder_logic(
  input [7:0] input_data
  output reg [15:0] decoded_output
);

localparam BIT_MASK_ONE = 8'h0F;
localparam SHIFT_ONE  = 4;
localparam BIT_MASK_TWO = 8'h3F;
localparam SHIFT_TWO = 2;


  always @(*) begin
    decoded_output[0:3] = input_data & BIT_MASK_ONE;
    decoded_output[4:9] = (input_data >> SHIFT_ONE) & BIT_MASK_TWO;
    decoded_output[10:15] = (input_data >> SHIFT_TWO + SHIFT_ONE) & BIT_MASK_ONE;
  end
endmodule
```
In this `decoder_logic` module the constants like `BIT_MASK_ONE`, `SHIFT_ONE`  `BIT_MASK_TWO` and `SHIFT_TWO` are localparams. They're only valid inside this module and you can’t change their values from the outside. They help you keep your expressions concise and readable without having to retype `8'h0F` or the value of shifts every time. It's mostly for internal organization and making your code easier to maintain.

One time I messed up and declared something as a `parameter` that should have been a `localparam`. I spent way too long trying to figure out why my designs were behaving all weirdly until I realized some other developer was accidentally overriding one of my parameters that I intended to be a fixed value.

To summarize if you want a constant value that you intend to be configurable on a module use a parameter. Use localparam if the value is supposed to be constant and only used inside that specific module.

**When to use which?**

Here is a bit more advice on when to use them.

*   Use `parameter` when:
    *   You want to create reusable modules with adaptable sizes or behaviors.
    *   You want to customize a module's functionality when you instantiate it in another module.
    *   You want to make it easy to experiment with various configurations of your module.
    *   You want to define constants that might need to change between instances or builds of your project.

*   Use `localparam` when:
    *   You need a constant value that's specific to the module you're working on.
    *   You want to create more readable code by assigning names to constants instead of scattering magic numbers.
    *   You want to organize internal logic of a module without exposing internal implementation details that other module might accidentally change.

**Some other nuances**

It's not all black and white. Sometimes it can get a bit murky. You can also use `parameter` inside a package to create a shared configurable value that you use in multiple modules which is an advanced technique when you are designing a big IP and you need to maintain all parameters the same throughout the design. It goes further than a localparam but it is another kind of constant.

Here is a package example:

```verilog
package my_pkg;
parameter MEM_WIDTH = 16;

endpackage

module ram_wrapper #(parameter ADDR_WIDTH = 10)
  (input logic[ADDR_WIDTH-1:0] addr,
  input logic[MEM_WIDTH-1:0] data_in,
  output logic [MEM_WIDTH-1:0] data_out);

 import my_pkg::*;
  //.. RAM implementation with the configured parameters ADDR_WIDTH and MEM_WIDTH
endmodule
```

Also it's important to know that `parameter` and `localparam` can also be used inside module interfaces and module classes in SystemVerilog, though with slightly different rules.

**Final Thoughts**

My rule of thumb? I usually default to using `localparam` unless I know I need a parameter. It keeps my module's internals more organized and I avoid unintended changes coming from another part of the design. And when in doubt I think if the value needs to be modified from outside the module then a parameter is a good choice. If not make it a localparam.

When I was a rookie I used to get them mixed up all the time. But after a few debugging sessions you'll start to get the hang of it. It’s like trying to figure out if a semicolon is needed in a C++ statement or not sometimes you forget but you learn from the debugging messages.

I would recommend checking out "SystemVerilog for Verification" by Chris Spear or "Digital Design and Computer Architecture" by David Harris and Sarah Harris. These books go over this topic in depth. I don’t recommend using websites like “allaboutcircuits” it is not a serious engineering source. Instead check the IEEE standard of verilog it should be available on ieee.org for a technical and precise explanation.
Hope this helps! Let me know if you have other questions.
