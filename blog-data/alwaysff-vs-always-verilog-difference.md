---
title: "always_ff vs always verilog difference?"
date: "2024-12-13"
id: "alwaysff-vs-always-verilog-difference"
---

Okay so you wanna know about `always_ff` vs `always` in Verilog right been there done that got the t-shirt let me tell you this isn't just some syntax sugar its a whole different way of thinking about hardware description trust me I've debugged enough of these things to have a few gray hairs so lets get down to brass tacks

Alright so lets talk about `always` first the classic the bread and butter of Verilog been around forever really its like that old reliable screwdriver you have in your tool box it works but sometimes it kinda gets you into trouble you know? The `always` block is a procedural block meaning things inside it happen sequentially its like a recipe you follow step by step you can use it for all sorts of things combinational logic sequential logic really whatever you throw at it that's both its strength and its weakness too much flexibility can make your code a bit of a mess if you're not careful see here's a basic example of an `always` block implementing some simple logic

```verilog
module always_example (
  input  wire clk,
  input  wire reset,
  input  wire a,
  input  wire b,
  output reg  out
);

  always @(posedge clk or posedge reset) begin
    if(reset)
      out <= 1'b0;
    else if (a & b)
      out <= 1'b1;
    else
      out <= 1'b0;
  end

endmodule
```

Okay so that code above right It uses an `always` block to define a register called `out` This register gets set to 0 on reset and then only when both inputs `a` and `b` are high will the register be set to 1 on the next clock edge other wise it stays at 0 See how easy that is? Now that specific example is fairly straightforward but the `always` block allows a lot of flexibility and sometimes people get a little too creative and that's where you can run into problems.

Now lets get to the cool kid on the block `always_ff` I mean it practically screams "flip-flop" doesn't it? This one is specifically for describing synchronous sequential logic your good ol flip-flops and registers nothing else I cannot stress this enough when you start using `always_ff` you're making a very explicit declaration to the synthesis tools "hey this is a flip-flop deal with it" that gives them a lot less room for interpretation and often results in more predictable and efficient hardware generation The tools know exactly what you intend to do they don't have to guess.

See `always_ff` came along in Verilog 2001 I think or maybe it was 2005 anyway it's meant to make your life easier by restricting the scope of what an `always` block can do when we're talking about register based logic it removes a lot of ambiguity about whether or not the logic inside is combinational or sequential and makes your intentions abundantly clear. Here is an example of the same functionality as above but using `always_ff`

```verilog
module always_ff_example (
  input  wire clk,
  input  wire reset,
  input  wire a,
  input  wire b,
  output reg  out
);

  always_ff @(posedge clk or posedge reset) begin
    if(reset)
      out <= 1'b0;
    else if (a & b)
      out <= 1'b1;
    else
      out <= 1'b0;
  end

endmodule
```
See it's pretty similar right? but the important detail is that with `always_ff` the tools know that this is a flip-flop and they aren't going to try and turn it into something else by mistake that might happen with the vanilla `always` block and trust me it can get ugly real fast I once spent 3 days tracking down a bug that was caused by me accidentally creating a latch using an `always` block because I missed one single case in the conditional it was my own fault obviously but I would have been spared the suffering had I been disciplined enough to use the `always_ff` keyword.

Ok so you might be asking ok yeah they seem similar why should I go to the trouble of using the new one when the old one works? Well the most important thing here is that using `always_ff` promotes *correct by construction* hardware design what do I mean by that? well it means that it forces you to think about your hardware design in terms of flip-flops and registers when your intent is actually to implement a flip-flop or a register no combinational logic inside a always_ff block It prevents you from making common mistakes that can lead to timing problems or unintended latches in other words its like using seatbelts while you drive.

Here's another example with a little more stuff going on its a shift register.

```verilog
module shift_register (
  input wire clk,
  input wire reset,
  input wire shift_in,
  output reg [7:0] data_out
);
  always_ff @(posedge clk or posedge reset) begin
      if (reset)
          data_out <= 8'b0;
      else
          data_out <= {data_out[6:0], shift_in};
  end
endmodule
```

In this example we are using the `always_ff` block to describe an 8-bit shift register. It shifts the input `shift_in` one bit at every clock rising edge and moves all bits one step to the left. This is a pure flip-flop based operation and it's a perfect use case for the `always_ff` construct The tools knows exactly what kind of thing it should generate in the hardware world.

So the key difference it really boils down to intent and the resulting restrictions. `always` is the general purpose workhorse you can throw all kind of logic at it but sometimes that makes it hard to keep track of exactly what you are making which is not always desirable in complex hardware and if you are a junior engineer well good luck debugging that you might end up wanting to become a software engineer by then I have seen that a few times. `always_ff` on the other hand is a specialist it is designed for synchronous sequential logic and it has specific rules that reduce ambiguity it forces you to write code that is easier to understand and synthesize. Its like the difference between a swiss army knife and a scalpel both tools are useful but one is more focused and more precise in its intended usage.

Now a little tech joke for you there was a programmer who was asked why did he start coding he said it's because I wanted to see if computers could count higher than me ok lame joke I know sorry but now back to what I was saying I swear I'll never do that again.

For learning more I would suggest looking into some books rather than randomly clicking on blog posts for that I'd suggest "Digital Design and Computer Architecture" by David Money Harris and Sarah L Harris or "Computer Organization and Design" by David A. Patterson and John L. Hennessy both of those are very solid books with tons of good information. "FPGA Prototyping by Verilog Examples" by Pong P. Chu is also a good choice if your main focus is FPGA development. Look for literature that is peer-reviewed you would be surprised how many people give advice that is totally wrong out there in the internet so be careful with your choices.

Anyway that's my take on it hopefully it clears things up for you and remember when in doubt always err on the side of clarity use the right tools for the right job and in most cases when you are dealing with flip flops or registers then use the always_ff because it is intended for that it will save you a lot of time and a lot of headache in the long run. Happy coding.
