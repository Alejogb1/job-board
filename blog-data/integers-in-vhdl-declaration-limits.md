---
title: "integers in vhdl declaration limits?"
date: "2024-12-13"
id: "integers-in-vhdl-declaration-limits"
---

Okay so integers in VHDL declaration limits right Been there done that got the t-shirt multiple times And yeah its a pain in the butt sometimes Lets break it down I've wrestled with this particular gremlin more than I care to admit Let's keep it straightforward

First off VHDL integers theyre not like the integers in some other languages You dont have this infinite range thing going on Youre constrained by the number of bits you dedicate to representing that integer During synthesis this translates directly to hardware resource usage so its pretty important to get right And its not a matter of simply changing a data type without thinking

The default integer type in VHDL is actually a *signed* integer think of it like a two's complement representation So if you don't specify anything it automatically goes for that which usually means a 32 bit range Which is great but what if you need more? or less? Youre gonna need more control than that And less if your FPGA resources need to be used correctly This is where things get interesting

So the key thing to understand about integer declaration is the range You gotta explicitly state the upper and lower bounds That bounds the number of bits allocated to the integer variable You usually see something like this

```vhdl
signal my_counter : integer range 0 to 255;
```

This declares a signal called `my_counter` as an integer that can hold values from 0 up to 255 This means your synthesized hardware would allocate an 8-bit register or a bunch of flip flops behind the scenes to hold that counter data And that's important If you accidentally exceed 255 it would overflow in a hard to debug manner And the debugger might not even show it

Now here's where I messed up back in the day I thought *integer* was just a convenient way to represent numbers during development I assumed the synth tool would do some sort of clever optimization But nope The tool will create hardware exactly based on what you specify if you have a range from -100 to 100 it's going to make those ranges for you even though you might not even reach those values and you might have some space to optimize

I was working on a video processing pipeline and I had a bunch of internal counters I was being lazy about declaring integers range because you know "oh ill optimize it later" big mistake I had all the counters running on a default 32bit integers even though they barely passed a maximum of 200 And you could imagine that wasted a lot of resources and I got hit with the *resource utilization too high* error during implementation Big slap in the face

Another common issue i had was trying to perform arithmetic operations that resulted in values outside of the specified range When you do that VHDL will not simply clamp the value or wrap it around you get undefined behavior during simulation and synthesis I spent hours tracking down a bug only to realize that a counter I had been incrementing reached its maximum value and just went bonkers Which brings us to overflow

Let's say you have this setup:

```vhdl
signal counter : integer range 0 to 7;
counter <= counter + 1;
```

What will happen after counter reaches 7 and increments? well this counter will just overflow and its not really deterministic you will get a random value instead of rolling over back to zero Which is definitely not what you usually want You need to actively detect overflow to prevent undefined behavior and do things like roll over in a loop or signal that you had an error

If you are using a counter like this and you try to read values after the counter reaches its maximum value you will probably get a random number and that's really hard to debug If you want rollover behavior you have to implement that logic yourself using modulus or if statements In my experience doing it right the first time is much less painful and leads to less debugging

The real learning point here is that VHDL integers are more like fixed-width data types than the dynamic integers you get in software I learned that the hard way

So to handle the different needs that I had back in the day I had to get very familiar with subtypes Subtypes allow you to create a new type based on existing ones with additional constraints For example you can make a subtype that adds ranges to integers or create an alias of sorts

Heres an example of how subtypes can be used

```vhdl
subtype small_int is integer range 0 to 15;
signal my_small_signal : small_int;
```

Here `small_int` is now a subtype that only allows values between 0 and 15 And I could reuse that definition through my code in a single place This makes the code much more readable and easier to maintain and you can clearly see the bounds in which that variable works The other way would be creating a `constant` and then reusing that `constant` variable and this is also an option depending on the complexity of your code

Another thing to be aware of is that range specifications have direction and it can be really confusing especially for people starting with vhdl For example both of these declarations are valid in VHDL but they represent different ranges

```vhdl
signal my_range1 : integer range 0 to 7;
signal my_range2 : integer range 7 downto 0;
```

While they both have same range of possible values 0 1 2 3 4 5 6 7 the way they are declared can have an effect on how synthesis tool will interpret those variables especially when creating indexed arrays And that got me once or twice

One more mistake I did was using the VHDL standard library `integer` type for doing math calculations and I never specified a range for the variable used to store intermediate results which leads to very hard to track bugs especially when multiplication operations are involved and this is why you should specify the limits when possible

Okay so a quick story so I can break up the text wall Back in the day I had a very weird bug where my whole design failed randomly at very specific frequencies and after days of debugging i realized it was because the integer variable I was using for my PLL calculations didn't have enough bits to properly perform the multiplication and it was wrapping around without any indication causing the whole system to fail at seemingly random times so yeah integers in vhdl are dangerous so be careful and specify the range whenever you can

Now if you really need some arbitrary integer representation that is bigger than 32 bits and you have no way to constrain it you might want to look into using `std_logic_vector` for large unconstrained bitvectors or create your own custom type and logic for it It makes the code more complex but you might need to do this depending on your particular situation I never had to do it but I can see the benefit depending on your application

When it comes to resources for more detailed information I would recommend the VHDL Language Reference Manual its dry but its the source of truth but that is sometimes like reading legal documents and it might be hard to understand if your are new to vhdl. For practical and more accessible information books like "FPGA Prototyping by VHDL Examples" by Pong P Chu are great source to understand common problems with VHDL including the integer problem and he is good at breaking it down into practical examples. I also recommend “VHDL for Engineers” by Kenneth Short its slightly more advanced but has very good discussion about the VHDL language itself and not just the synthesis and gives very good explanations on why VHDL is the way it is.

In short specify the range when declaring integer signals its essential for efficiency and avoids nasty bugs. Its not like a software language where integers are infinite use the smallest number of bits you need that is safe from overflow when doing your calculations and operations and always think about your physical limitations of your FPGA board and always be aware of your hardware resources and plan your logic based on that limitation
