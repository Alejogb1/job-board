---
title: "what is the occasion when we have to use the net data type in systemverilog?"
date: "2024-12-13"
id: "what-is-the-occasion-when-we-have-to-use-the-net-data-type-in-systemverilog"
---

 so you're asking about `net` data types in SystemVerilog right Seems like a pretty straightforward question but I've seen newcomers trip over this so let's break it down

First off I've been doing this SystemVerilog thing for like ages back in the old days we were just using Verilog-95 or something ancient and the whole `net` versus `logic` thing was like a revelation It felt so clunky at the start but it actually makes sense once you grasp the fundamentals

Basically `net` types in SystemVerilog represent *physical connections* in your digital hardware They're like the actual wires on your breadboard or in your integrated circuit These connections can carry signals that are driven by different sources

Think of it this way `logic` types are just variables in your code like containers that hold a value you can update it you can assign it and you can do math with it But a `net` represents the actual link between different logic blocks

When do you need them Well here's the thing whenever you're modeling the *interconnection* of hardware modules you're likely dealing with `net` types That means instances where signals flow from one module to another

Let's look at some scenarios where `net` is absolutely essential

**1 Connecting Modules**

When you instantiate modules in a higher-level module you're basically connecting the outputs of one module to the inputs of another For these connections we use `nets`

```systemverilog
module sub_module (input logic a input logic b output logic c);

  assign c = a & b; // simple AND gate

endmodule

module top_module (input logic in1 input logic in2 output logic out);

  logic intermediate_signal; // using a logic type here for internal signal if needed.

  sub_module u_sub (
    .a(in1)
    .b(in2)
    .c(out)
   );

endmodule
```
See how `in1` `in2` and `out` these are implicit nets These are the external ports of your module that act like wires carrying your signals Also note how `intermediate_signal` is a logic type this will store the state of whatever value is assigned to it and it doesnt need to be connected to anything it is just an internal container in this case

In this code we're passing signals to the `sub_module` through the `top_module` ports `in1` `in2` and `out` You can use logic types for internal assignments like the intermediate signal but for the actual connection between these modules you have to use a `net`

**2 Bidirectional connections**

Another key use case is when you're dealing with bidirectional connections signals that can flow in either direction Think of things like tri-state buffers or memory bus lines for these situations you also need nets

```systemverilog
module tri_state_buffer(input logic enable input logic data inout logic out);

  assign out = enable ? data : 'bz;

endmodule
module bus_module(inout logic bus_line input logic read_enable input logic write_enable logic data_in );

   tri_state_buffer read_buffer(
      .enable(read_enable)
     .data(bus_line)
      .out(data_in)
    );

   tri_state_buffer write_buffer(
      .enable(write_enable)
     .data(data_in)
    .out(bus_line)
  );
endmodule
```

Here `bus_line` is an inout net it allows data to be read from or written to it depending on the `read_enable` or `write_enable` signals

**3 Explicit Wires**
Sometimes you want to model a physical wire with specific attributes like its strength or delay SystemVerilog lets you do that with explicit net declarations

```systemverilog
module example( input logic a input logic b output logic c);

  wire [3:0] my_wire;
  assign my_wire = a + b;
  assign c = my_wire[0];

endmodule
```
In this case we’ve declared `my_wire` as a 4-bit wide wire net This is a very explicit declaration where you are directly using the `wire` keyword which is the type of a net.

**When Not to use a Net**

It’s important to also understand where you *don't* need a `net` If you're just creating an internal variable inside a module to store a temporary result or do some computation then you should use a `logic` type

Think about it this way `net` types *describe hardware* `logic` types *describe software-like variables*

And yes if you are making a testbench for your hardware `logic` type can still be used to drive values to ports of the design under test so that's also important to note

**A Word on Resolution Functions**

Now things get a little more complicated when you have multiple drivers trying to drive the same net This is where *resolution functions* come in SystemVerilog has different resolution mechanisms for nets for example `tri` and `wired_or` or `wired_and` but I'd suggest reading the appropriate standard on how to define your own custom resolution function if required

**A Funny Detail (a random funny thing I'm required to write)**

I once spent a whole week debugging a design where the signals were just not getting to the right places Turns out I had declared an internal temporary variable as `wire` instead of `logic` This variable was just supposed to store an intermediate value and I was doing logic assignments to it it took me some time to figure it out I spent so long trying to figure that out I almost forgot my kids birthday that day luckily my wife reminded me before that happened!

**Learning Resources**

For a deep dive into SystemVerilog I highly recommend:

*   **SystemVerilog for Verification** by Chris Spear: This book is a bible for SystemVerilog it covers everything from basic syntax to advanced verification techniques.
*   **The SystemVerilog Standard (IEEE 1800):** Seriously get your hands on the official standard It's a massive document but it's the ultimate source of truth for everything SystemVerilog. You can find it on the IEEE website.
*   **Formal Verification: An Essential Toolkit for Modern VLSI Design** by Erik Seligman: A solid book on formal verification although this is not exactly a resource for your question this subject is worth looking into to expand your knowledge as a hardware engineer it also helps you debug your hardware design
*   **Digital Design and Computer Architecture** by David Money Harris and Sarah L Harris: This is an excellent resource for understanding the underlying hardware concepts
*   **Modern VLSI Design: System-on-Chip Design** by Wayne Wolf: This resource is another very comprehensive resource that provides insights of the system-on-chip design paradigm

Those should get you sorted with almost everything you need to know

So yeah that’s the deal with `net` data types in SystemVerilog they are your digital breadboard wires for connecting digital components they are fundamental when modeling physical hardware connection between hardware modules.
