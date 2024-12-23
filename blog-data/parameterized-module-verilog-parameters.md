---
title: "parameterized module verilog parameters?"
date: "2024-12-13"
id: "parameterized-module-verilog-parameters"
---

 so you're asking about parameterized modules in Verilog parameters right I've been there man oh boy have I been there Let me tell you about this one project back in my university days we were building this ridiculously complex ALU thing for a computer architecture course and yeah parameters in Verilog were our best friend but also our worst enemy initially

It all started so simple you know a basic adder module right I'm talking like this:

```verilog
module adder (input [7:0] a, input [7:0] b, output [7:0] sum);
    assign sum = a + b;
endmodule
```

Easy peasy lemon squeezy right?  8-bit adder done and dusted But then professor says " now make it a 16-bit adder" And then a 32-bit adder and then a 64-bit one I swear he was doing this just to torture us We were copy pasting modifying and breaking our fingers just to get every size working and it's not really efficient right Every time it's a new size a new module? That was not gonna fly

That's when parameters walked in and gave us a much-needed rescue. So here's how we made the thing parameterized

```verilog
module parameterized_adder #(parameter WIDTH = 8) (input [WIDTH-1:0] a, input [WIDTH-1:0] b, output [WIDTH-1:0] sum);
  assign sum = a + b;
endmodule
```

See that `#parameter WIDTH = 8`? That's the key We're declaring a parameter called `WIDTH` and giving it a default value of 8 This means if you don't specify a size when you instantiate the module it'll default to an 8-bit adder. And the inputs and outputs are all sized based on that `WIDTH` which is amazing

Now when you want a 16-bit adder you don't need to create a whole new module you can just do this:

```verilog
parameterized_adder #(16) sixteen_bit_adder (
  .a(input_a_16bit),
  .b(input_b_16bit),
  .sum(output_sum_16bit)
);
```

See we're passing `#16` when we're instantiating the module so the `WIDTH` is now 16 inside that instance. And that creates a 16-bit adder. We used names `.a()` `.b()` etc because it helps to prevent mismatches on wiring and also for clarity. I've seen people omit the names but sometimes it is hard to figure out what is what.

The real magic though started when we needed a ripple carry adder vs. a carry-lookahead adder We'd have different implementations for different architectures and needed to swap them in and out so to speak and parameters helped us create reusable components

 so you might be asking about the difference between `parameter` and `localparam` You see `parameter` can be overridden at instantiation time like above `localparam` on the other hand once defined is a constant it cannot be changed outside the scope. We used `localparam` for things like internal state machine definitions it gave better readability and safety. If you want things not to be accidentally changed in a given module `localparam` is what you want. It works like `const` in other programming languages

Let's go back to our adder story. So we had the basic adder but then we needed a subtractor too You can get it out of the adder with little tweaks, with addition of invert and proper handling of the carry bit. But we needed to control that logic using a new parameter

```verilog
module parameterized_alu #(parameter WIDTH = 8, parameter OPERATION = "add") (
  input [WIDTH-1:0] a,
  input [WIDTH-1:0] b,
  output reg [WIDTH-1:0] out
  );
  
  always @(*) begin
   if(OPERATION == "add") begin
     out = a+b;
   end
    else if (OPERATION == "sub") begin
      out = a-b;
    end
    else begin
      out = 0;
    end
  end
endmodule
```

See here we've added a new parameter called `OPERATION` and by default set it to `add`. So you can instantiate the ALU module with:

```verilog
parameterized_alu #(8, "sub") sub_unit(
  .a(input_a_8bit),
  .b(input_b_8bit),
  .out(output_sub_8bit)
 );
```

And now you have a sub unit it's kinda cool right?

We took it further with a `select` parameter to handle logic operations. It's like having all the logic bits available as LEGO blocks You know sometimes it is like doing plumbing. You spend 90% of the time figuring out the connection points 10% doing it.

I think I need a drink now. I've written way too much today.

 so here are some recommendations on resources that can help you further

*   **"Digital Design: Principles and Practices" by John F. Wakerly:** This book is a classic It's not super Verilog-specific but it gives you a really solid foundation in digital design concepts which is key to understanding *why* parameters are so useful in the first place.

*   **"SystemVerilog for Design" by Stuart Sutherland:** This goes much further into the more modern aspects of Verilog so after you grasp the basics you will want to look at this book.

*   **IEEE 1364-2005 Verilog Standard:**  this one is not fun to read but it is the specification that defines the standard of the language. If you really want to understand the deep details there is nothing better than the standard document itself.

I hope this is detailed enough and addresses your question let me know if you have other questions I will be happy to help
