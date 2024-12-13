---
title: "verilog functions usage modules?"
date: "2024-12-13"
id: "verilog-functions-usage-modules"
---

Okay so verilog functions and modules huh seen this one a few times definitely know the drill lets break it down and talk about how I've seen it play out in real life

First off functions and modules are both crucial but they are *very* different beasts It’s easy to get them confused especially when you’re just starting out I remember way back when I was designing my first FPGA based audio processor I was hitting my head against a wall because I was trying to use a function for something that clearly needed to be a module lessons were learned that day let me tell you

**Verilog Functions**

Think of verilog functions as the workhorses for repetitive tasks they are designed to perform operations that are combinatorial meaning that their output is solely a function of their inputs there is no memory or state involved they compute values pure and simple The key part is that a function *must* return a value they are not for making hardware that keeps states

Here’s a simple example of a function that calculates the sum of two 8-bit numbers

```verilog
function [7:0] add_numbers (input [7:0] a, input [7:0] b);
  add_numbers = a + b;
endfunction
```

Easy enough right This function takes two 8-bit inputs `a` and `b` adds them together and returns the 8-bit sum its super useful for making code more readable and less redundant say you are making a digital signal processing and you have to add several variables this small piece of code can be used several times and you are not copying and pasting code and the result is way more readable

Important things to know about functions

*   **Combinatorial Logic:** They can only contain combinatorial logic that is no `always` blocks
*   **Return Value:** They must have a return value declared in the function definition
*   **Called Inside `always` Blocks:** You typically call functions within `always` blocks or other functions but not on their own
*   **Local Scope:** Variables declared inside a function are local to that function only this is good practice and avoids problems with accidental overlapping variables names

**Verilog Modules**

Modules are the fundamental building blocks of hardware in verilog They represent a piece of hardware that can have inputs outputs and internal logic including state elements (registers or flip flops) They are used to create hierarchical designs think of them as bigger blocks that can encapsulate more than a function can they are the true power houses of verilog They also dont return anything they are designed to create pieces of hardware that communicate with others by their inputs and outputs

Here is a module that implements a simple flip flop register (it stores 1 bit)

```verilog
module simple_register (
  input clk,
  input reset,
  input d,
  output reg q
);

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      q <= 1'b0;
    end else begin
      q <= d;
    end
  end
endmodule
```

This module is much more powerful than a function it includes an `always` block a state element `q` and responds to clock edges and the asynchronous reset this type of design cannot be made with a function because functions cannot have an `always` block nor have the ability to respond to clock signals

Key takeaways for modules

*   **Hierarchical Design:** Modules can instantiate other modules allowing for complex design hierarchies
*   **Can contain state elements:** Modules can contain registers and other stateful elements using `always` blocks
*   **Concurrency:** Modules operate concurrently the logic inside them runs in parallel
*   **No return value:** Modules do not return values they manipulate signals and values inside of them or its outputs

**Usage Differences**

The core difference lies in their intended purpose functions are for combinatorial operations modules are for hardware blocks with state and concurrency

*   **Functions for computation:** If you need to perform calculations data manipulations or other purely combinatorial tasks you use functions It makes the code shorter and readable You should reuse your functions to prevent redundancy
*   **Modules for hardware:** For everything else you use modules registers flip flops memories counters all of those are modules they are hardware blocks you need to do things with

For example let's say you want to calculate a CRC-32 checksum on data you might write a function for the core CRC-32 calculation logic this CRC function would process the data but not store anything then you would use a module to manage the input data feed that data into the CRC calculation function and store the result This way you get a module with state that uses a function to make its work

**My personal example when functions where not enough and I needed a module**

Way back when I was working on a custom encryption engine on an FPGA I started off by trying to implement the core encryption algorithm as a verilog function I wanted to write as much as possible in functions so my code was more compact and readable it was faster to write too the first few lines I was happy with myself the idea was to take the data as an input use a function to process it and return the encrypted data

But I quickly hit a wall I needed to add state to the design to track the internal round counter of the encryption algorithm you know every encryption algorithm needs a round counter I needed a register and registers can not be made with a function because there is no `always` block to capture state it was a good time to learn that I was approaching the wrong direction

I realized the core operation of the encryption like the addition xor rotations and so on could be functions but the actual algorithm needed to be a module because it needed to have state and I ended up instantiating other smaller modules inside to manage all that logic

I was trying to use a spoon to dig a trench I needed a module

Here is a small example of instantiating the function inside the module

```verilog
module encryption_module (
  input clk,
  input reset,
  input [31:0] data_in,
  output reg [31:0] data_out
);

  reg [7:0] round_counter;

  function [31:0] basic_encryption_op (input [31:0] data);
    // Some basic operation like shift or xor
    basic_encryption_op = data ^ 32'h5a5a5a5a;
  endfunction

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      round_counter <= 8'd0;
      data_out <= 32'd0;
    end else begin
      if(round_counter < 16) begin
        data_out <= basic_encryption_op(data_in);
        round_counter <= round_counter + 8'd1;
      end
    end
  end
endmodule

```

In this example `basic_encryption_op` is a function and the module is where this function gets its state and its clocking and controls the round counter this function does the heavy lift of the computation but the module controls the overall process of this simple fake encryption algorithm

I learned my lesson that day and I haven't mixed up my functions and modules ever since the difference is very clear now

**Resources**

If you want to really dive deep into verilog I would not recommend some online tutorial they are usually very bad I would recommend "Verilog HDL" by Samir Palnitkar its a classic and provides very good understanding of the core principles or "Digital Design and Computer Architecture" by David Harris and Sarah Harris its a very good hardware and digital design book and touches on verilog in good detail they are both great resources for understanding the difference between functions and modules and a lot more in verilog

In conclusion use functions when you are making simple blocks for computation and modules for everything else that involves more hardware like flip flops registers counters memories and so on if you use functions when you need a module you will be staring at your computer frustrated like me that one day
