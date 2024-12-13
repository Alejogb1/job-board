---
title: "concatenate inputs in verilog?"
date: "2024-12-13"
id: "concatenate-inputs-in-verilog"
---

Alright so you wanna concatenate inputs in Verilog huh Been there done that got the t-shirt and probably a few scars from debugging late night synthesis runs let me tell you its a classic problem with a few solid solutions and some gotchas you should definitely know about

First off lets talk about the basics The easiest way to smash signals together is using the concatenation operator its that `{}` curly brace thing you see everywhere you can throw individual bits variables even whole vectors into it it’ll just line them up like they’re waiting for a bus kinda thing

```verilog
module concatenate_example (
  input [7:0] a,
  input [3:0] b,
  output [11:0] c
);

  assign c = {a, b};

endmodule
```

That simple `assign c = {a, b};` takes your 8 bit input `a` and your 4 bit input `b` and makes a new 12 bit vector `c` where `a` is in the most significant bits and `b` is in the least significant bits its dead simple right but dont go thinking that's the only way you can use this operation or that its always gonna be this straightforward now this gets much more useful when you want to put bits in a specific order or when some of the inputs are expressions not just simple variables

Lets say you have inputs from a bunch of modules each with different widths and you have to stuff them in a different way for example a data bus control signals and status flags you cant just concatenate in order because they would be all mixed up what you need is to have that ordered concatenations now if you get the order wrong your design will be messed up a common mistake so try to be meticulous here always check and recheck your order of the bits and signals

Here’s a example showing how you can combine different signals and also how to use literal values in the concatenation to create a specific format

```verilog
module complex_concatenate (
  input [15:0] data_bus,
  input [2:0] control_signals,
  input status_flag,
  output [22:0] packet
);

  assign packet = {2'b10, data_bus, 3'b000, control_signals, 1'b0, status_flag};

endmodule
```
notice how we mixed variables and binary literals inside the curly braces we created a fixed header `2'b10` added the data bus some zeros as padding the control signals and an end of packet bit and a status flag that’s a common technique when creating a data packet or a memory address

Ok now where things get hairy is when you start using the replication operator inside the concatenations its the `{n{signal}}` thing that repeats the given signal n number of times so it’s not just adding it like `assign c = {a,a,a};` in the previous examples it is the `n` amount of copies of signal what’s the big deal you ask well it looks simple but you can make your design look much cleaner without creating helper variables specially if you need the same value repeated more than just 3 or 4 times also this is a useful feature that reduces the amount of code you need to write and makes it more readable trust me on this your future self will thank you

I had to debug a micro controller bus interface a while ago that heavily used this replication I had to stare at the waveform for hours before realizing I just had to count how many times they replicated the address bus but yeah you live and you learn and you keep the documentation open for this kinds of things now after that I always create helper variables for the replicated signals when it has a lot of replicas because that is how I learn not to suffer in the future

```verilog
module replication_concatenate (
  input [3:0] data,
  input parity_bit,
  output [31:0] output_data
);

  assign output_data = {{8{parity_bit}}, {4{data}} , data, 4'b0000};

endmodule
```
In this example we are creating an output data packet we replicated parity bit 8 times added 4 times the data then added the data and finally four zero bits for good measure think of this as filling up a memory address with a particular pattern if you want to understand it

Now for a gotcha that will probably bite you in the butt one day and I mean it the sizes of the inputs in the concatenation need to be well defined its a huge mistake to concatenate signals with unknown sizes that happens usually when using parameters that are not defined or if your inputs are using non constant width signals it is very common and when it happens it gives you weird synthesis errors or weird waveform behaviors I mean we're talking about unexpected signal values or timing issues in the design and if you are not careful with this kind of situation you might fall into the famous situation where you spend all your day debugging something really stupid that you should have avoided from the beginning

This is why good documentation and verification are important for those cases and I cannot stress this enough documentation and testing is a must here for this kind of situation always specify the width of your signals no matter how obvious it may seem it makes it easier for you and others in your team to understand

So lets talk about synthesis for a second and what kind of optimizations does synthesis tools do with concatenation operations in a simple case like `assign c = {a, b};` you wont see much logic generated because it is just a wiring connection between a and b it is just a new vector made of `a` and `b` but if you start throwing a lot of replication and weird bit selections the synthesizer might choose to use multiplexers or other logic to create the signal you want

Remember that synthesis tools are optimizing for resources and they might not always generate the exactly logic you expect in a simple example but they will always do what you intended even if sometimes its in a more optimized way but that’s mostly a good thing right you would want your hardware to be efficient that is why we use Hardware Description Languages in the first place otherwise we’d be drawing schematics with transistors

Another thing you should take in to account is the timing of your signals in synchronous circuits a typical concatenation will not add much delay it is just a wire but if you are using replication or very wide vectors it might need a little bit of time to propagate through the hardware and you need to account for that and that is why you should always run timing analysis after implementation of your designs to make sure your design is within the time constraints of your hardware

Alright now lets talk about resources because I am a big fan of books and papers instead of random tutorials on the internet

You definitely need a good verilog reference book I recommend "Verilog HDL" by Samir Palnitkar its a classic and goes deep into everything including concatenation It is useful for a good foundation for everything you might need to know about Verilog syntax and use cases it is a staple in many universities

For a more advanced understanding of the synthesis implications and how these operations get translated into hardware I would recommend "Digital Design and Computer Architecture" by David Money Harris and Sarah L Harris it covers the fundamentals of digital design but also talks about the hardware implementations and what happens under the hood of a synthesis tool so that you understand why the synthesizer tools will make the optimization choices they do it also gives you a solid base for design principles and best practices which is always a good thing for a hardware engineer

Another important resource is the IEEE standard for Verilog that is the IEEE 1364 standard that defines everything about the Verilog language it is kind of boring and is a pain to read but it is necessary to have it open in case of any doubts or confusions with specific operations or if you are in the mood to delve into the nitty gritty details of how it works

So summing up remember the basics of concatenation the replication operation the gotchas with signal sizes and consider the synthesis and timing implications also always read the documentation and verify your design is always important no matter how simple the design is Also use those books and standards it’s not optional its crucial for a good and reliable design and with that you should be concatenating inputs in Verilog like a pro and without issues that you cannot fix
