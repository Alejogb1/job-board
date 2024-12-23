---
title: "passing parameters between verilog modules?"
date: "2024-12-13"
id: "passing-parameters-between-verilog-modules"
---

 so you're asking about passing parameters between Verilog modules right Been there done that more times than I can count It's like the bread and butter of hardware description but also a source of constant headaches if you don't get it right

Let me tell you this was definitely not my first rodeo. Back in my early days when I was still wet behind the ears I spent probably a week trying to debug a communication interface because of a parameter mismatch. I had a bus controller and an arbiter and for some reason data was just gibberish. The issue turned out to be a bit-width mismatch between a parameter declared in one module and the one used in the other that I didn't see at first. I had a deep sigh moment when I found it. You know how it is the simplest things make you go crazy.

So yeah I've seen this problem from every angle imaginable So lets get to it We have several ways to handle parameters in Verilog

First and foremost you need to understand that parameters are constants defined within a module that can be changed during module instantiation. This is very different from wires or registers whose values can change during run time. You might think of them as compile-time settings for your module which really is a great way to think about them.

The most basic way to pass parameters is by directly overriding the default values during module instantiation using the `#()` syntax. This is probably the first way you encounter when learning Verilog. It’s very readable and straightforward you just need to remember the syntax. For example if you have a module like this:

```verilog
module multiplier #(parameter WIDTH = 8)
(
  input  [WIDTH-1:0] a,
  input  [WIDTH-1:0] b,
  output [2*WIDTH-1:0] out
);
  assign out = a * b;
endmodule
```

You can instantiate it with a different `WIDTH` like this:

```verilog
module top;
  wire [15:0] res1;
  wire [31:0] res2;

  multiplier #(.WIDTH(8)) mult1 (
      .a (8'hAA),
      .b (8'hBB),
      .out(res1)
  );

   multiplier #(.WIDTH(16)) mult2 (
      .a (16'hAAAA),
      .b (16'hBBBB),
      .out(res2)
  );
endmodule
```

Here we are creating two `multiplier` modules one with a `WIDTH` of 8 and the other with 16 You see how that works? Notice the dot before the parameter name that's the important part. If you forget that or have an extra space it's not going to work you should pay attention to that.

Now let's say you have a hierarchy of modules and you need to propagate parameters through multiple levels. It’s very common in real-world designs believe me I've been there. You will be creating a hierarchy of modules that are dependent on each other and at some point a module deep into your design needs to change depending on an input parameter.

For example you have this module:

```verilog
module inner_module #(parameter INNER_WIDTH = 4)
(
    input [INNER_WIDTH-1:0] data_in,
    output [INNER_WIDTH-1:0] data_out
);
  assign data_out = data_in;
endmodule
```

And an intermediate module that uses it:

```verilog
module intermediate_module #(parameter INTER_WIDTH = 8)
(
    input [INTER_WIDTH-1:0] data_in,
    output [INTER_WIDTH-1:0] data_out
);

    inner_module #(
        .INNER_WIDTH(INTER_WIDTH)
    ) inner (
        .data_in(data_in),
        .data_out(data_out)
    );

endmodule
```

And then the top module:

```verilog
module top_module;
    wire [15:0] data_out_top;
    intermediate_module #(.INTER_WIDTH(16)) intermediate (
        .data_in(16'h1234),
        .data_out(data_out_top)
    );
endmodule
```

Here `top_module` uses `intermediate_module` which in turn uses `inner_module`. The parameter `INTER_WIDTH` is passed down from the top and sets `INNER_WIDTH` inside the inner module so that both parameters are linked to the same value set by the top instantiation. It's like a domino effect. I found this way of coding the hierarchy very clean and practical I think you should also use this method.

The other way is to use the `defparam` statement this one is less preferred and in modern Verilog code this is rarely used. It is considered bad practice and creates a spaghetti code style because it can be used anywhere to modify parameters which leads to readability issues. It can still be useful sometimes and it's important to know how to use this one. It works by changing the value of parameters during runtime but that can lead to problems in terms of synthesis. Its not something you should be proud of to write code like this but hey let me show you. I'm not advocating it but just for completeness.

```verilog
module mod_a #(parameter A = 10)(output reg [A-1:0] out);
  always @* out = 0;
endmodule

module mod_b;
  reg [7:0] out_b;
  mod_a mod_a_inst ();
  defparam mod_a_inst.A = 8;
  assign out_b = mod_a_inst.out;
endmodule
```

Here we are setting `A` parameter of module `mod_a_inst` from module `mod_b` to 8 after the module was instantiated. This is generally less readable and harder to manage. I don't recommend it at all but sometimes you will see it in old legacy code bases and this is why it's good to know this.

A little funny anecdote is that once I accidentally used `defparam` in a very large design and I literally changed parameters of a module that was being used in different places of the design which in turn was creating a very weird behavior. It took me a very long time to debug this one. The moral of the story don't use `defparam` unless you really know what you are doing and you cannot use another method. It's like using a chainsaw to cut butter. It works but it's not elegant nor recommended.

So that's essentially how you deal with parameters in Verilog. The important takeaway is understanding the purpose of parameters and passing them correctly through your design hierarchy. This is a skill that comes with time and experience.

I would highly recommend reading the SystemVerilog standard IEEE 1800-2017. It includes the language constructs for both Verilog and SystemVerilog. Also the "Digital Design and Computer Architecture" by Harris and Harris is a good resource for digital design and RTL coding concepts. I find them quite useful when I started working with Verilog and hardware design.

Remember to keep it simple always test your code and pay attention to the details. Happy coding!
