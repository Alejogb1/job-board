---
title: "verilog ternary operator usage examples?"
date: "2024-12-13"
id: "verilog-ternary-operator-usage-examples"
---

Alright so you're asking about Verilog's ternary operator huh Been there done that got the silicon burn scars to prove it This thing's a real workhorse if you know how to tame it Otherwise it can leave you scratching your head at 3 AM debugging a seemingly simple logic block Trust me I've been there

Let's break it down the ternary operator in Verilog is basically a compact way to write an if-else statement it's syntax is this condition ? true_expression : false_expression It evaluates the condition if it's true it returns the true expression otherwise the false expression Pretty straightforward right?

But like with any tool understanding its subtleties and limitations is key to using it effectively I've seen so many juniors and even some seniors make simple mistakes using this because it can be too terse and if they do not write proper comments the code becomes a hell to understand

The first time I seriously ran into trouble was working on a pipelined RISC-V processor way back I was trying to implement a simple branch prediction unit I thought I'd be clever and use a whole bunch of ternary operators to determine the next PC address based on the branch outcome Man was that a disaster in waiting First thing was timing issues due to the depth and complexity of that nested ternary expression it was a critical path nightmare the synthesis tool choked and the hardware I ran into weird corner cases that were impossible to trace The moral of that story is don't go overboard nesting these things keep it simple readable or your debugging sessions will be endless

So let's look at some actual examples lets start with the most basic one here:

```verilog
module mux_2to1 (
  input wire a,
  input wire b,
  input wire sel,
  output wire out
);

  assign out = sel ? a : b;

endmodule
```

This snippet is the most basic example it is a two-to-one multiplexer If sel is high then it outputs a and if not it outputs b Its equivalent to this:

```verilog
module mux_2to1_if_else (
  input wire a,
  input wire b,
  input wire sel,
  output wire out
);

  always @(*) begin
    if (sel) begin
      out = a;
    end else begin
      out = b;
    end
  end

endmodule
```

As you can see the ternary operator version is more concise but the if-else version is also more explicit It depends on the context what is better to use The key thing to notice that the signal out in the first snippet does not use always @(*) begin because the assignment is done directly by the ternary operator This avoids potential inferring a latch which can lead to very hard-to-debug problems

Now lets move a bit more complex imagine you are working on an ALU and need to select between different operations based on a function code you will use a case statement but for simple cases the ternary operator could be your friend

```verilog
module alu_operation (
  input wire [3:0] func_code,
  input wire [7:0] operand_a,
  input wire [7:0] operand_b,
  output wire [7:0] result
);

  assign result = (func_code == 4'b0001) ? operand_a + operand_b :
                  (func_code == 4'b0010) ? operand_a - operand_b :
                  (func_code == 4'b0011) ? operand_a & operand_b :
                  (func_code == 4'b0100) ? operand_a | operand_b :
                                         8'bx; // Default case (undefined output)
endmodule
```

In this example we use nested ternary operators to decide between addition subtraction and bitwise operations This shows how you can chain them but again note that this can get very messy and error prone very fast if you have too many conditions If the func_code is 0001 it adds operands if its 0010 it subtracts and so on Notice that the last case outputs an x this is very important to define a default value because synthesis tools could generate a latch and cause headaches later I personally prefer to avoid these constructions for more than 2 layers deep otherwise is better to use case statements

One important thing to remember is that the ternary operator requires all expressions on both the true and false side to be of the same data type or the compiler will throw an error This tripped me up when I was doing my master's project I was trying to assign a register to a wire without proper type casting It took me like half a day to fix that bug just because I did not check the errors carefully

Finally remember that unlike some other programming languages Verilog doesn't allow you to use the ternary operator directly on blocks of code only on single assignments for control flow you are better of using the if-else construct because you cant put code blocks on each condition this is important because it limits a bit how powerful it can be

There is one more area I can mention that is when you are using a case when you are using tristates in some architectures where you have a common bus this was very common in older systems although still useful today

```verilog
module tristate_buffer (
  input wire enable,
  input wire data_in,
  output wire data_out
);

  assign data_out = enable ? data_in : 1'bz;
endmodule
```

Here the ternary operator allows you to select between data\_in and high impedance state when enable is low the output becomes high-Z when is high it forwards the data This is a standard trick for bus sharing

One thing I really hate to see is juniors using ternary operators without understanding the underlying hardware I've seen way too many cases where the synthesis tool generated a complex mux tree when the simple if-else construct would have been more efficient They do it because they heard that ternary operators are faster which is not always the case it depends on the type of hardware and the synthesis tool

Also please use a simulator It's amazing how many bugs a simple test bench can catch If you're serious about hardware design I suggest diving deep into papers on hardware synthesis optimization there is one good book that is "Synthesis and Optimization of Digital Circuits" by Giovanni De Micheli it explains many of the algorithms that synthesis tools implement and how you can get more efficiency

I once wrote a very complex expression with tons of ternary operators It took me hours to debug that it was not that complex of a problem but the fact that I was tired and the fact that I did not have a good testbench made me waste a whole day debugging it So the lesson here is test your code do a simulation plan and do not be an idiot like me and do not use the ternary operator without thinking twice it is useful but also dangerous

Remember the key here is to write clean and readable code not just compact code use the right tool for the job and dont be afraid to use a normal if-else construct if that is more readable than a huge messy ternary expression

Oh and one last thing why did the Verilog program get a bad grade because it was full of latches Haha Sorry I had to sneak that one in there

Anyways I hope that helps you Good luck with your Verilog projects and may your logic never glitch
