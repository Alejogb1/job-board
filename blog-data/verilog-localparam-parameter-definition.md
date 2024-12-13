---
title: "verilog localparam parameter definition?"
date: "2024-12-13"
id: "verilog-localparam-parameter-definition"
---

Okay so Verilog localparam parameter definitions huh Been there done that got the t-shirt several times actually Let's dive in because this is a rabbit hole many have fallen into and I've clawed my way out more than once

First things first when we talk about `localparam` and `parameter` in Verilog we're essentially talking about constants We're setting up fixed values that can be used throughout our design Think of them as named magic numbers that make your code way more readable and maintainable The key difference is scope and how they can be overridden

A `parameter` is like a public constant It's declared in a module but it can be overridden by instantiating modules This means when you use a module in a larger design you can change some of its internal parameters from outside the module itself Think of it like a customizable module with knobs you can turn when you use it That's powerful but sometimes you want constants to be just that constant

That's where `localparam` comes in This guy is a private constant It's defined within the module and its value cannot be changed externally It's like a secret ingredient that the module keeps to itself and it’s set in stone You can't mess with it from outside You’ve probably guessed that that can be quite important in maintaining consistency and preventing accidental changes within your designs This all comes down to design hierarchy and controlled data flow you know like proper engineering practices or something

Here's a simple Verilog example to show what I’m talking about

```verilog
module my_module #(
    parameter DATA_WIDTH = 8 // This is a parameter can be overwritten during module instantiation
)(
    input [DATA_WIDTH-1:0] data_in
    output [DATA_WIDTH-1:0] data_out
);

  localparam HALF_WIDTH = DATA_WIDTH/2; //This is a private constant internal to this module

  assign data_out = data_in  ^ {HALF_WIDTH{1'b1}}; //xor with a constant depending on internal parameter

endmodule
```

In this example we have `DATA_WIDTH` which is a parameter that can be modified when this `my_module` is used as part of larger design For example you can instantiate it like this:

```verilog
module top_module(
    input [15:0] top_data_in
    output [15:0] top_data_out
);

 my_module #(
   .DATA_WIDTH(16) //override the original value of 8 to 16
) my_instance (
    .data_in(top_data_in),
    .data_out(top_data_out)
);

endmodule
```

Notice how we overrode `DATA_WIDTH` to 16 in `top_module` via module instantiation In contrast `HALF_WIDTH` inside the `my_module` remains fixed It’s determined by `DATA_WIDTH` within the module definition but you can't change its final value by instantiating it I think of this like a sealed box once `my_module` is compiled `HALF_WIDTH` will be the same every time you use `my_module`

Here's another scenario where `localparam` shines especially when it comes to derived constants

```verilog
module another_module #(
 parameter BASE_ADDRESS = 32'h0000_1000
)(
    input logic [31:0] addr,
    input logic data_in,
    output logic data_out
);

  localparam OFFSET = 32'h0000_0004;
  localparam FULL_ADDRESS = BASE_ADDRESS + OFFSET;

  assign data_out = (addr == FULL_ADDRESS) ? data_in : 1'bz; // example usage
endmodule
```

In this case `OFFSET` and `FULL_ADDRESS` are derived from parameters but remain constant once `another_module` is compiled You can't override `OFFSET` or `FULL_ADDRESS` from outside `another_module` by design This type of construction is super useful to ensure addresses or other offset values do not get messed up

Now let me tell you a story about a design I worked on back in the day where I didn’t use `localparam` appropriately Picture this a huge FPGA design involving multiple modules a complex data pipeline It was a nightmare of epic proportions Everything was parameterized which initially felt good but it was a mess I had like a parameter called `BUFFER_SIZE` which was used everywhere And I was trying to debug a nasty data corruption problem and turned out some of the modules got the wrong `BUFFER_SIZE` and it was overwriting memory in other modules. All because I forgot to fix the instantiation parameters during one of the many iterations of the design You think it is funny now? I did not then it was hell that's how I learned about the power of localparam for derived constants. I could have made a derived size within the modules and be sure nothing messed up with that constant and make my life easier

So `localparam` and `parameter` are both powerful tools in Verilog The trick is knowing when to use which one

Here's my golden rule of thumb if a constant should be user-configurable (overridden from outside a module) use a `parameter` If the constant is internal to the module should not be tampered with or is derived from a module's `parameter` use a `localparam` Simple right? I know I know you might say this is the least of your problems but these things often bite you back at the worst moment

A common mistake I see is people trying to use localparam for constants that should change per module instance Like for address mapping that needs to be customized It is a very very common mistake so don't feel bad if you have done that as well Just learn from your mistakes

So yeah localparam its super useful to ensure internal constance and `parameter` to make your modules configurable. It is not rocket science but if you get it wrong you will have a bad time

Also remember when defining parameters and local parameters you do not need to give them a value if you are going to change them or define them later

```verilog
module another_module #(
    parameter BASE_ADDRESS // no value assigned
)(
    input logic [31:0] addr,
    input logic data_in,
    output logic data_out
);

    localparam OFFSET //no value assigned yet
    localparam FULL_ADDRESS = BASE_ADDRESS + OFFSET; //used offset before it has been assigned? not allowed!

  assign data_out = (addr == FULL_ADDRESS) ? data_in : 1'bz; // example usage
endmodule
```

You can add initial values later or through a different file or configuration It is a bit advanced but useful so keep it in mind

Where to learn more? Well you can always read the official Verilog standard its dense but useful and a good thing to read at least once but I would recommend:

*   **"Digital Design and Computer Architecture" by David Harris and Sarah Harris:** This book explains the principles of digital design and how hardware description languages fit in it's not verilog specific but it explains the concepts in a way that will help you use `parameter` and `localparam` more effectively.
*   **"FPGA Prototyping by Verilog Examples" by Pong P. Chu:** This is a practical guide that focuses on Verilog and provides real-world examples so you will understand how to use these concepts in actual implementations It's more hands-on.
* **"SystemVerilog for Verification: A Guide to Learning the Testbench Language" by Chris Spear:** This one is more specific to verification but it explains parameters and local params in SystemVerilog very well which has many of the concepts of verilog so it can help to understand the concepts well

Remember use `parameter` for configurable constants and `localparam` for internal private constants It's about controlling what can be changed and how so keep these rules in mind and you will be all set
