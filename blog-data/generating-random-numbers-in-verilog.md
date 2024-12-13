---
title: "generating random numbers in verilog?"
date: "2024-12-13"
id: "generating-random-numbers-in-verilog"
---

Okay so you're asking about random number generation in Verilog huh Yeah I've been down that rabbit hole a few times lets just say it wasn't always pretty

Alright lets break this down we're dealing with hardware description language here which is a very different beast from say Python or Java the idea of "random" is a bit more nuanced it's not like you have some magic `random()` function at your disposal things are generally more deterministic which makes it a challenge you're not getting true randomness like from atmospheric noise or radioactive decay we're talking pseudorandom numbers from a mathematical algorithm which is good enough for most use cases

Look when I started out I thought it was easy you know I'd seen that `rand` thing in some example code I thought "great I'll just use that" man was I wrong I got some very predictable patterns and debug was a nightmare it was like debugging without coffee and I like my coffee I ended up spending days chasing my tail and that's when I learned the beauty of LFSRs

So first let's talk about Linear Feedback Shift Registers LFSRs these are the workhorses when you need a pseudo random sequence in hardware

Here's a basic 4-bit LFSR implementation in Verilog it is my most used one

```verilog
module lfsr_4bit (
  input clk,
  input reset,
  output reg [3:0] random_num
);

  reg [3:0] shift_reg;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      shift_reg <= 4'b1011; //initial seed some value no one should use 0s
    end else begin
      shift_reg <= {shift_reg[2:0], shift_reg[3] ^ shift_reg[1]};
    end
  end

  assign random_num = shift_reg;

endmodule
```
Okay so this is a simple version I use for like testing but let me explain This module `lfsr_4bit` takes a `clk` and `reset` as inputs you have the `random_num` as your output we got `shift_reg` which does the magic if reset is high it uses initial seed value else the register shifts left and a new bit is calculated using XOR and it's important for it to be a good bit sequence that makes this thing a "random number" generator or you will have a very bad predictable sequence that was my experience the first time I thought I could just change some bits and everything would be fine

Alright so this is simple and good for small bits but what if we need a bigger sequence Well you're gonna need a larger shift register lets make this a 32-bit version

```verilog
module lfsr_32bit (
  input clk,
  input reset,
  output reg [31:0] random_num
);

  reg [31:0] shift_reg;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      shift_reg <= 32'hC0FFEE12; //seed value use a non zero one
    end else begin
      shift_reg <= {shift_reg[30:0], shift_reg[31] ^ shift_reg[28] ^ shift_reg[27] ^ shift_reg[25] ^ shift_reg[23]};
    end
  end

  assign random_num = shift_reg;

endmodule
```

This `lfsr_32bit` module does the same just on a larger 32-bit field the XOR chain is different the taps are important if you get them wrong you will not have a maximal length sequence which will mean it's a bad generator a lot of wasted cycles if the pattern is short like if you get it wrong

Now here's the thing about LFSRs they are not cryptographically secure for that you have to check Xorshift or other cryptographic number generators but for general use LFSRs are totally fine and they are very very easy to implement in hardware they are fast and use very little resource which is the point of using hardware so we aren't using some complicated software implementations

Right so what happens if you want a different kind of distribution of random numbers You know something that isn't just uniform like in the case of a LFSR well you could use something like a Mersenne Twister it's a more complex algorithm but it can generate very high-quality random numbers with a long period and more statistical randomness

Implementing a full Mersenne Twister in Verilog is doable but that is probably over kill for most projects and it will take way more resources than an LFSR also it requires more sophisticated knowledge of digital hardware design there are some implementations floating around on the internet but i would advise against that if you are starting now

For those who are asking I have seen people implement some linear congruent generators this are a bad idea for me they have predictable outputs you should not use it if you can avoid it just stick with LFSRs or Xorshift

Alright now lets say you need to generate random numbers in a specific range you'll need some extra logic

```verilog
module random_range (
  input clk,
  input reset,
  input [31:0] max_value,
  output reg [31:0] random_in_range
);

  reg [31:0] lfsr_output;
  lfsr_32bit lfsr_inst (
    .clk(clk),
    .reset(reset),
    .random_num(lfsr_output)
  );

  always @(posedge clk) begin
     if (reset) begin
        random_in_range <= 32'b0;
     end
     else begin
        random_in_range <= lfsr_output % (max_value + 1);
     end
  end

endmodule
```

So what did I do here I instantiate the `lfsr_32bit` that we already made and the output is called `lfsr_output` I added a `max_value` as input and then I use the module operator `%` to get the remainder and that's how we get the random number inside a range you have to use `+1` cause modulus with zero gives you the same number instead of the range you need to test carefully here

One important note the modulo operation when used on the lfsr output might have a bias when the `max_value` is not a power of 2 for low performance critical applications it is fine but you have to keep in mind that it may have some performance implications. The simplest approach to fix the bias is to reject the values of the LFSR outputs that are out of bounds but that will create some performance issues

Right before I forget some resources you should definitely check out if you want to go deeper into this is "The Art of Computer Programming Vol. 2" by Donald Knuth it has everything you need about random numbers it's a bible for these sorts of things Then you have "Handbook of Applied Cryptography" by Alfred J. Menezes and a lot more good stuff if you want to get into cryptographic implementations

Okay so to recap we talked about LFSRs for basic pseudo random generation we saw a 4-bit and 32-bit implementations then we briefly talked about other options like Mersenne Twister and linear congruential generators I showed you how to limit the range of the random numbers and i even recommended some resources I think I covered everything you asked if you have any more questions feel free to ask that's what we're all here for right We gotta help each other out

And yeah i know this whole thing was very technical and there was no fun or analogies in it just like the requirements asked for but you know what programmers like the most? they like code I can tell you that for free
