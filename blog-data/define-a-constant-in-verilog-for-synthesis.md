---
title: "define a constant in verilog for synthesis?"
date: "2024-12-13"
id: "define-a-constant-in-verilog-for-synthesis"
---

 so you wanna define constants in Verilog for synthesis right I get it This is like Verilog 101 but with a twist because you're asking about synthesis not just simulation I've been there man trust me Been there done that and got the T-shirt well not literally a T-shirt but you get the idea

So let's talk about constants in Verilog for synthesis I mean you can declare constants a bunch of ways but for synthesis it's not always straightforward Like you can't just use any old variable willy-nilly

First off the most basic way is using the `parameter` keyword This is probably what you'll use most of the time It's like the bread and butter of Verilog constant declarations It allows you to define a value that's fixed during synthesis but it can be overridden when instantiating the module which is pretty neat

```verilog
module my_module #(
  parameter WIDTH = 8,
  parameter DEPTH = 16
) (
  input  logic [WIDTH-1:0] data_in,
  output logic [WIDTH-1:0] data_out
);

  // Use the WIDTH and DEPTH parameters within the module
  logic [WIDTH-1:0] my_buffer [0:DEPTH-1];

  // ... some other logic ...

endmodule
```

See that `WIDTH = 8` and `DEPTH = 16` those are parameters They're constants They're fixed for a given synthesis run unless you override them from outside the module when you instantiate it

This way you can make your module flexible but the parameters are still constants from the module's perspective within itself If you try to change `WIDTH` inside the `my_module` you're gonna get a synthesis error and rightfully so

Now parameters are like the user interface of a module They provide a way to configure it when you wire it into a larger design They are the constants that can be modified before hardware is generated think of it like template arguments in C++ They allow you to create generic modules and instantiate them in a variety of different sizes and configurations So `parameter` keyword is your go to friend usually

Another way to create a constant is with the `localparam` keyword This is similar to `parameter` but it can't be overridden when you instantiate the module It's like a constant that's completely local to the module It's a bit more rigid and less flexible compared to `parameter` which is a good thing when you want to enforce an internal fixed value that should not be changed by any outside module

```verilog
module my_module (
  input  logic [7:0] data_in,
  output logic [7:0] data_out
);

  localparam MY_CONSTANT = 10;
  
  logic [7:0] counter;

  always_ff @(posedge clk)
  begin
  counter <= counter + MY_CONSTANT;
  end

  // ... some other logic using MY_CONSTANT ...
endmodule
```

In that example `MY_CONSTANT` is a `localparam` it's set to 10 and that's it No changing it from the outside when you instantiate `my_module` This is like if you wanted to have a buffer with a specific fixed size that is not modifiable from any higher level in the system

`localparam` is the go-to when you have internal constants that must not be changed from outside the module So if you have a constant that is really local and for internal use within the module then use the `localparam` keyword and you'll be fine

 here's a thing I ran into once This was a nightmare let me tell you it's not about using `parameter` or `localparam` but about how you use them in more complex situations

Let's say you're doing some kind of DSP processing and you need a bunch of pre-calculated constants for a look-up table You cannot just fill your source code with random numbers and magic constants We were debugging a complex system for a week before we realized the issue It was a huge mistake We could have avoided it if we used parameters or localparams correctly

The problem was that some engineers hardcoded the numbers instead of using parameters it was like "oh this specific LUT size is 127 so let me just write 127 directly into the code" I know right like why? It's insane and as you can imagine it became impossible to make changes quickly because there were numbers spread all over the place It was horrible

Here's how you should use parameters for these LUT types of things

```verilog
module lut_module #(
  parameter LUT_SIZE = 256
)(
  input  logic [$clog2(LUT_SIZE)-1:0] address,
  output logic [7:0]                 data_out
);

  logic [7:0] lut [0:LUT_SIZE-1];
  
  initial begin
   for (int i = 0; i < LUT_SIZE ; i = i+1)
    begin
      lut[i] = i % 255;
    end
  end

  assign data_out = lut[address];

endmodule
```

In that snippet `LUT_SIZE` is a parameter you can change it before the synthesis stage if you instantiate the module and the rest of the LUT logic is automatically adjusted based on the new size and you do not have to hard code 256 anywhere in the module so it makes it more readable and more maintainable

The other really cool thing with parameters in synthesis is that in many cases the synthesis tool can use the values to do a bunch of optimizations so if the parameter is set to a very small value the synthesis tool can decide to use fewer logic resources to implement your module or even if the constant is a power of two it can make certain optimizations regarding resource usage

Also use `$clog2` system function that returns the ceiling of the log base 2 of the value It's quite useful for figuring out the minimum number of bits to represent a number which is handy for things like address widths and stuff I use `$clog2` all the time like in the example above

 so here comes the part of the question which requires a joke but not really a funny one It should still be related to tech though. I'm gonna try to make it sound like someone said it on stackoverflow.

Why do Verilog programmers prefer dark themes? Because light attracts bugs ha ha ha.

Anyway moving on from the dad joke So just keep in mind that the way you declare constants impacts not just how your code is understood but also how the synthesizer will work to generate the physical implementation of your circuit

 a few things that are useful to know but not strictly related to defining constants but are related to the use of constants in synthesis:

*   **Avoid complicated constant expressions in synthesis** It's fine for simulation to have complex expressions with parameters but for synthesis try to keep things as simple as possible Synthesis tools can struggle with really complex expressions sometimes leading to synthesis failures or very poor area/timing tradeoff
*   **Use descriptive names for your constants** This sounds obvious but sometimes people are lazy and just write some random constant names use names like `NUM_OF_BYTES` instead of just `NUM` or `WIDTH` instead of `W`. So use descriptive constant names it will make your life much easier in the long term
*   **Use comments to explain your constants** It's always a good practice to comment on all your code but it's especially useful for documenting why you have chosen a particular value for your constants which is not always obvious from the name alone
*   **Consider using Verilog packages** For constants that are used across multiple modules or for common design constants use packages to make your life easier so you don't have to rewrite constants all over the place again and again

so here are some resources that I think are very good and I recommend a lot These are more in depth and have more technical explanations of what you should expect when working with constants in hardware description languages such as verilog

First off definitely read the **"SystemVerilog for Design" by Sutherland et al.** This book is a classic for learning SystemVerilog which also includes Verilog and it has a very detailed chapter on parameters localparams and constants also there is a lot about synthesis considerations for constant related code. Its a must have resource

Secondly check out **"Digital Design Principles and Practices" by John F. Wakerly** This book is more about digital design in general but it does have good coverage of the hardware implications of choosing particular data widths and constant values. It will help you understand why certain constant values are faster or better in terms of area of your digital circuits

Finally read some papers like the ones that are often published by ACM Transactions on Design Automation of Electronic Systems (TODAES) especially those that discuss hardware synthesis methodologies and how parameters and constants can affect final hardware implementations it's a bit more academic reading but its always good to get a deeper understanding of these more in depth issues

I've spent many years debugging things that could have been avoided by defining constants correctly from the get go so trust me this stuff is important It's not just about making your code compile it's about making it efficient maintainable and bug-free so you do not get to the debugging hell that we have all experienced in our careers

So remember `parameter` for flexible constants `localparam` for fixed internal constants and don't be afraid to define your constant in a descriptive way Also keep in mind synthesis optimizations and read your books and papers. And maybe try to avoid hardcoding things in your designs It makes your life very difficult trust me on that one

 that's about it If you have any other questions I will be glad to help I try to check stackoverflow whenever I can. Good luck with your Verilog projects hope this helps.
