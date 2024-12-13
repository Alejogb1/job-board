---
title: "bitwise or in verilog circuit?"
date: "2024-12-13"
id: "bitwise-or-in-verilog-circuit"
---

Okay so bitwise OR in Verilog yeah seen that rodeo a few times lets break it down real simple and I’ll throw in some stuff I’ve learned the hard way

First things first bitwise OR is a fundamental logic operation it takes two or more inputs and outputs a 1 if *any* of the inputs are 1 otherwise it outputs a 0 It’s about as basic as it gets in digital logic circuits You'd think it’s a no-brainer and it usually is but there are some things to be mindful of especially when you start thinking about actual hardware implementation and all that entails.

In Verilog you use the `|` operator for bitwise OR simple enough right? So if you have two signals like this:

```verilog
module or_example(
    input [3:0] a,
    input [3:0] b,
    output [3:0] result
);

assign result = a | b;

endmodule
```

This code defines a module `or_example` that takes two 4-bit inputs `a` and `b` and outputs a 4-bit result. The `assign` statement is where the bitwise OR happens. The `|` operator is applied bit-by-bit meaning `result[0]` gets the OR of `a[0]` and `b[0]` `result[1]` gets the OR of `a[1]` and `b[1]` and so on. Pretty straightforward when you just use it like that.

Now one thing that got me into trouble early on was not paying attention to the size of the operands. Let’s say you have something like this

```verilog
module size_mismatch(
    input [7:0] longer_signal,
    input smaller_signal,
    output [7:0] result
);

assign result = longer_signal | smaller_signal;

endmodule
```

Here `longer_signal` is 8 bits and `smaller_signal` is 1 bit. What happens in this case? The Verilog simulator will usually throw a warning saying that operands of different sizes are being used in the bitwise OR operation. It will effectively pad `smaller_signal` with zeros on the left so in effect you’ll be ORing `longer_signal` with `8’b0000000x` where x is the bit value of smaller signal. If you are not aware of this this can cause very strange unexpected behaviour. If you are trying to only affect one bit in a longer bus by doing bitwise OR you should take explicit actions to do so this is the first rule of thumb. You know you shouldn’t let the simulator do things for you without being aware of it. This leads to so many errors down the line. Explicit is always better in hardware design.

So you might be thinking ok I get the basic bitwise OR thing but when does it really become useful? well everywhere! It's really important for things like setting bits in registers or combining control signals for different modules. For example if you have various interrupt sources and each is represented by one bit then you can OR them together to create an interrupt signal that’s high if *any* of the sources have raised an interrupt. I’ve had so many projects fail because of incorrect interrupt handling early on trust me on this one. There are lots of good textbooks on Digital systems design that are very useful if you intend to do this.

Let's say you are working with a memory mapped register which I've done in plenty of projects before It's very common in embedded systems. Let's imagine a control register where bit 0 is an enable bit and bit 1 is a reset bit and you want to set these bits individually without touching the others. You could do that by this type of code which works fine for a simple case.

```verilog
module register_manipulation(
    input [1:0] enable_reset_input,
    input clk,
    output reg [7:0] control_register
);

always @(posedge clk) begin
  if(enable_reset_input[0]) //Enable bit 
     control_register <= control_register | 8'b0000_0001; // set bit 0
  if(enable_reset_input[1]) //Reset bit
     control_register <= control_register | 8'b0000_0010; // set bit 1
end
endmodule
```

In this module the `enable_reset_input` represents the incoming bits you want to write to the register with each bit of the input correspomding to a given bit in the register The crucial part is this line: `control_register <= control_register | 8'b0000_0001;` This line takes current value of `control_register` and bitwise OR's it with `8'b0000_0001` effectively setting the least significant bit while leaving the others unchanged. Similarly the other `if` block does the same with `8'b0000_0010` setting bit 1. I had to fix so many bugs because of this register manipulation and I made my fair share of it too. So this is a good practice to use this often.

Now a little trick I have seen in the wild but be careful when you use it is if you have a signal which you want to use to gate another signal but you want to make sure the signal to be gated is also high even when the gating signal is not high for a very short duration. You can OR both signals together that means the gated signal will only be low when both signals are low. It’s a little hacky but can sometimes be useful when you know what you are doing but use it only when you are sure of your timing requirements. So for example lets say you have `request_signal` and `grant_signal` and `request_signal` has a short pulse and you want to use `grant_signal` to indicate to the system that `request_signal` has arrived but you still want to see the `request_signal` pulse even if the `grant_signal` is low or inactive you can do this.

```verilog
module signal_gating(
    input request_signal,
    input grant_signal,
    output reg gated_signal
);

always @* begin
 gated_signal = request_signal | grant_signal;
end
endmodule
```
In this example I am using an always block to implement combinatorial logic as well and that works too for many simpler modules like this but it can be problematic if the block is complex. That `gated_signal` will be 1 if either `request_signal` or `grant_signal` is 1 so you will still have the `request_signal` if `grant_signal` is 0 but it will also show the presence of `grant_signal`. This also is very prone to creating timing issues so you should use this type of logic with care. I once made an entire chip malfunction due to a misunderstanding of timing issues so that gave me a bad case of headache and a real appreciation for static timing analysis.

So yeah bitwise OR it looks simple at the surface level and it is but it’s fundamental to many many parts of digital design. You need to understand the underlying hardware implications such as the gate delays and understand that simulators are not always the best representatives of what happens on real silicon or an FPGA but they are still extremely crucial for hardware design. Always pay attention to operand sizes be explicit about what you are doing and don't let the simulator make any implicit conversions without your knowledge. I tell my team the same thing all the time I think I am going to get a t-shirt saying be explicit in hardware design! And of course always double-check your bit manipulation because bit-level errors can be very hard to find! Oh and if you are wondering I get my knowledge mostly from IEEE journals and old textbooks on digital systems design such as those by Katz and Mano. These textbooks and resources have been invaluable. I have learnt so much from them more than anything else to be honest.
