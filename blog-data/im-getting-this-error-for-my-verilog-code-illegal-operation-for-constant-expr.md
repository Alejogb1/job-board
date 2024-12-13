---
title: "i'm getting this error for my verilog code illegal operation for constant expr?"
date: "2024-12-13"
id: "im-getting-this-error-for-my-verilog-code-illegal-operation-for-constant-expr"
---

Okay so "illegal operation for constant expr" in Verilog huh Been there man countless times Let me tell you its like a rite of passage for anyone serious about hardware description languages It's usually something stupid you overlooked and you are going to hate yourself for it after figuring it out

Okay so the "constant expression" bit its referring to something that should be resolvable at compile time The Verilog compiler wants to figure out this calculation during the compile phase not during the simulation or hardware execution phase This is because constant expressions are used in a lot of places like parameter definitions or array dimensions or even case statements and well if the value is not known at compile time then the compiler throws a tantrum And rightfully so

The "illegal operation" part is what really boils your blood It's not specific in its error message because the Verilog standard has a ton of operators and not all of them are fit for constant expressions You can't expect the compiler to execute every single piece of code before the hardware even runs It expects that a good number of things can be precalculated So we need to figure out what exactly is causing this

Been there myself I remember this one time when I was working on this custom encryption engine back in my university days I was a bit cocky with my understanding of Verilog I decided I'd implement this modular inverse calculation as part of a parameter definition because you know why not I tried to do something like this

```verilog
module encryption_engine #(
    parameter MODULUS = 101 ,
    parameter INVERSE = (MODULUS**-1)%MODULUS
 ) (
    input clk ,
    input data_in ,
    output data_out
 );
  // Rest of the module code
 endmodule
```

Okay so this seems alright at first right Well its not because the power operator `**` in Verilog is not synthesizable unless the exponent is a constant integer. You can't calculate a modular inverse like this during compile time The compiler looked at this and just chuckled and gave me the error message you are facing

The correct fix obviously was that I had to precalculate that value and use a const value instead Or if it was crucial to do it on the fly I had to create an actual module for it.

Here’s another example and this one hits close to home because it got me stuck for hours one Friday night Let me tell you what I was doing

I was trying to create a generic memory module and I thought I was super clever using parameters to define the memory size and address width because I am very professional obviously

```verilog
module memory #(
    parameter DATA_WIDTH = 8 ,
    parameter DEPTH = 2**(DATA_WIDTH/2) , //Error Here
    parameter ADDR_WIDTH = $clog2(DEPTH)
  ) (
    input clk ,
    input  [ADDR_WIDTH-1:0] addr ,
    input  [DATA_WIDTH-1:0] data_in ,
    input  we ,
    output [DATA_WIDTH-1:0] data_out
 );
  reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

  //rest of the module
endmodule
```

The error is in the `DEPTH = 2**(DATA_WIDTH/2)` line because the result might not be an integer if data width is odd and well Verilog doesn't exactly handle that very well for parameters Its not a good way to do it for sure It needs an integer and thats a basic rule The fix is to ensure that the result is an integer so use an integer variable or a round function or something that doesn't depend on runtime calculations.

But the real slap in the face here is the `$clog2` which is only acceptable if the value you pass it is constant And since `DEPTH` is computed during compilation it's not a constant for the purposes of `$clog2` even if it looks that way so another compilation error right there. The compiler has no chill.

This made me learn to take each step more methodically so I could avoid such errors.

Here's yet another example this one simpler

```verilog
module example #(
    parameter A = 5,
    parameter B = A + 10*A
   ) (
  );
  //rest of the module
endmodule
```

This looks fine too right? But say I add a bit of complexity which is extremely common in hardware coding:

```verilog
module example #(
   parameter A = 5,
   parameter C = 7,
   parameter B = (A * C + 10 * A )
   ) (
   );
   // rest of the module
endmodule
```

Here I'm assuming that `*` multiplication is a compile time constant operation and it is but what if for some reason I am using a custom library that redefines the multiplication operator and uses some crazy non constant time algorithm to do so. Well this would cause an error because the compiler assumes that a `*` operation is a constant operation but my custom library might not be operating in the same way or using the same logic hence we get an error because it is now an illegal operation for a constant expr because `B` would rely on something computed during simulation time and not compile time which is what parameters need to be

The error message is not very helpful I know but when you know the root cause it starts making sense so my suggestion is to always verify your expressions and make sure you are only using constant operations

The rule of thumb is that anything that's used to define a parameter array size or case statement needs to be pre-calculated And when I say precalculated I mean either a literal number like 5 or a simple expression using parameters that are simple integers and that do not use operators that may be synthesizable but might not be fit for compile time calculations.

The fix for these kinds of errors depends on the context but usually involves

1.  **Pre-calculating values** If something is based on some crazy calculation do it yourself on your calculator or use a script if you need to.
2.  **Use only constant-valid operators** Not every operation works as a constant expression in verilog so double check and make sure that your operations are safe for constant evaluation
3. **Careful with `$clog2`** Remember that argument to it needs to be a compile time constant too so it might be tricky sometimes.

For a more in-depth look I'd recommend these resources they've helped me a lot over the years

*   **"Digital Design and Computer Architecture" by David Money Harris and Sarah L Harris**: This book explains a lot about digital logic design in general and you will find the hardware design side of things very useful it teaches you to think like a hardware designer which is extremely important.
*  **"SystemVerilog for Verification" by Chris Spear**:  Though it focuses more on verification SystemVerilog its a great book with good explanation of the details that may help you write better code and avoid common issues.
*  **The IEEE 1800-2017 SystemVerilog Standard:** You know when things are too confusing this is what you read its the final reference and it contains every possible rule and operation. Its dense and you might need a few years experience to navigate through it but once you understand it you really understand it. I know that nobody will read it but honestly this is the best source.
*   **"Verilog HDL" by Samir Palnitkar** The classic textbook on Verilog that explains everything from the basics to advanced stuff.

Debugging these things can be a bit of a pain. You might also want to try to use a better IDE. I remember when I started writing verilog I was using a plain text editor now a good IDE can catch these errors before you even compile the code so it might help. It is like getting a free linting tool. You need all the help you can get and these tools can reduce a lot of time in debugging things.

Also try to break down your complex expressions into smaller pieces and see exactly where the error is originating and remember that the compiler is your friend it is not trying to annoy you it is trying to tell you something. Okay I know I am probably lying on that last bit but what else can I say.

Remember to always double-check your constant expressions and make sure everything is resolvable at compile time.

Keep coding and don't let those error messages get you down. We all have been there. I mean the error is so common people think they found a bug in the compiler and they start sending emails to the company saying "hey you got a bug" its not a bug its you buddy!

And remember its not always the tools fault sometimes we need a bit more coffee a few more hours of sleep and a calm mind to tackle these issues.

And one last tip before I let you go I know you are eager to get back to coding just remember that parameters are supposed to be compile time constant and the compiler is not going to execute the most complex operations for you.

Ok that's all I have for now. Good luck.
