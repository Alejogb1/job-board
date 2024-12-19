---
title: "difference among always ff always comb always latch and always?"
date: "2024-12-13"
id: "difference-among-always-ff-always-comb-always-latch-and-always"
---

Okay so you're asking about Verilog's always blocks right Yeah I've been there done that burned my fingers more than a few times let me tell you This is one of those things that looks simple on the surface but can bite you hard if you're not careful Especially when you're dealing with timing and synthesis results ugh

First thing first forget any notion of always being just "always" in a regular software sense It's not a loop that just runs continuously Instead it’s about how the hardware simulator interprets your logic and how it translates to physical circuits

Let’s break it down you've got four common types of always blocks always ff always comb always latch and plain old always

**`always @(posedge clock)` or `always @(negedge clock)` or `always @(posedge clock or negedge reset)` for example is always ff**

This one is your bread and butter for sequential logic the `ff` is short for flip flop You'll hear that term a lot in hardware design Flip flops are the fundamental building blocks for things like registers counters and state machines The posedge clock or negedge clock bit that's the key This means the code *inside* the always block only gets executed when the clock signal transitions either from low to high (posedge) or from high to low (negedge)

If you’re not familiar with clock signals they are like a heartbeat for your digital circuit Everything that needs to happen synchronously is triggered by it It's a rhythmic tick-tock that keeps all the different parts of your circuit working together

Why use always ff? Because it enforces a nice time-based behavior you can easily predict This avoids the sort of timing disasters that happen when you start introducing combinational logic everywhere things will change faster than what the physical hardware can handle It's great for stability and is the cornerstone for almost every sequential logic module ever designed

Here's a quick example:

```verilog
module d_flipflop (
  input  logic d,
  input  logic clock,
  output logic q
);

  always @(posedge clock) begin
    q <= d;
  end

endmodule
```

See that `<=`? that's a non-blocking assignment in Verilog it says that on the next clock cycle the value of `d` will be assigned to `q` This is the right way to do it inside clocked always blocks

**`always @(*)` is always comb**

This is the second most common one after the previous one and it’s used for combinational logic `comb` is short for combinational which means logic that reacts instantly to input changes no clock involved This is things like multiplexers decoders encoders adders simple gates and many many other pieces of hardware that calculate based on inputs

The `@(*)` means that the block is triggered by *any* change on any of the input signals used inside that block If you were to write down every signal that influences the output you will find they are included when you use `*` so you don't need to think much about it it does it for you

Now here is a bit of a history lesson back in the old days we had to explicitly list *every* input signal in the sensitivity list of the `always @()` block It was like herding cats if you forgot one your simulator might show something different than your real hardware especially if you were testing it outside the simulator This `@(*)` notation is a lifesaver and a gift from the hardware gods if you ask me. A sanity savior if you will

Here’s an example:

```verilog
module two_to_one_mux (
  input  logic a,
  input  logic b,
  input  logic sel,
  output logic out
);

  always @(*) begin
    if (sel)
      out = b;
    else
      out = a;
  end

endmodule
```

Here the output `out` changes immediately whenever either `a` `b` or `sel` changes Now that is fast right?

**`always @(some_signal or some_other_signal)` with level sensitivity is always latch**

Okay now we are getting into trickier territory The always latch or level sensitive always block This one is easy to misunderstand if you aren't careful You usually have a explicit sensitivity list as well and it can be triggered by level changes in the signal instead of edges like in flip flops

These are used for things like generating a clock gate where a change in level is used as control signal or latches for memory controllers but mostly used to create latches (a type of memory storage element similar to flip-flop). The level sensitivity means the code executes as long as the input signal is active (like a enable signal) and in most cases this is what defines a latch. Remember these latches are often inferred by mistake and in most cases this is considered bad design style

Here is an example

```verilog
module gated_latch (
  input logic d,
  input logic enable,
  output logic q
);

  always @(enable or d)
    if (enable)
        q = d;

endmodule
```
In this case whenever enable goes high or goes low it updates the output with the value of d it basically stores that value. In most cases this is bad design because these types of logic are really not what you expect to be doing in a always block

**The plain old `always` block**

Now when you have just `always` by itself without the `@()` this is what causes most of the confusion and headaches

This type of block can be interpreted in several ways depending on the synthesizer you're using It can get really unpredictable and most likely will not work in your design It's often used in simulation to create what is called a testbench that doesn't need to be synthesized since it doesn't model any hardware It can also be used for procedural code that you would like to run always like a task that you can call and run different times during your simulation

Here is a example of how you might see in a testbench

```verilog
module testbench;
    reg clock
    initial begin
        clock = 0;
        forever #5 clock = ~clock; // Simulate a clock signal forever
    end
    // Other testbench code here using clock
endmodule
```

In summary always ff is for clocked sequential logic always comb is for combinational logic always latch is level sensitive usually for latch implementation and the plain `always` block is used for simulations

**My Hard Lessons Learned**

I remember once I tried to use a plain always block to control some kind of data processing chain The simulator did something completely unexpected at that time it seemed to work I synthesized it and of course the chip I was working with did not behave as I had simulated It was a timing nightmare the circuit was a big mess of unwanted latches that I didn't even know that where there

It took me days of debugging with a logic analyzer to find out what I did and how I made such a bad mistake I had mixed sequential logic with combinational logic and got really weird timing issues The fix was easy using always ff with some registers and a proper clock but the hard lesson is etched in my memory forever It is not a problem I want to repeat ever again

So always ff is your best friend for anything related to timing always comb for all the things that calculate output immediately and always latch is for level sensitive things that you should not use on your main design and for `always` without `@()` run far away as quickly as possible if you plan on synthesizing hardware. If you want more details on the inner workings of hardware synthesis I suggest reading “Synthesis of Digital Circuits” by John P Hayes this book was a lifesaver in my early days I would also recommend reading “Digital Design and Computer Architecture” by David Money Harris and Sarah L. Harris that also covers more design topics for hardware engineers. I hope this helps and be careful with those always blocks it’s really easy to make a mistake
