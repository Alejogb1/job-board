---
title: "concatenation in vhdl signals code?"
date: "2024-12-13"
id: "concatenation-in-vhdl-signals-code"
---

 so you're asking about concatenating signals in VHDL right Yeah I've been there done that got the t-shirt probably spilled coffee on the keyboard doing it a few times too Lets dig in shall we

I've seen so many fresh VHDL devs trip over this I mean I tripped over it when I was starting out its a rite of passage I guess Back in my early FPGA days I was working on a custom image processing pipeline you know the kind that needs insane amounts of parallel processing I had all these separate pixel data paths for different color components red green and blue each represented by an 8 bit signal Now I needed to combine them into a single 24-bit signal for further processing or to send it off to a display controller and of course me being the genius I was initially tried to manually bit-shift and add them yeah disaster area code looked like a drunk snake

The issue you're hitting or might hit is that VHDL uses the `&` operator for concatenation It's not like adding them numerically or something its about gluing bit vectors together end to end think of it as shoving Lego blocks together and building a bigger wall not adding them like you would in math or something

Here’s the basic gist lets say you have these

```vhdl
signal red_data : std_logic_vector(7 downto 0);
signal green_data : std_logic_vector(7 downto 0);
signal blue_data : std_logic_vector(7 downto 0);
```

You'd get your concatenated RGB data like this

```vhdl
signal rgb_data : std_logic_vector(23 downto 0);
rgb_data <= red_data & green_data & blue_data;
```

Bam simple as that That would give you a 24-bit vector named rgb_data where the high bits are the red low bits are blue and green sits in the middle You know the usual RGB way you see in all the textbooks. Order matters of course if you swap the order it will all be different so it depends what you want to achieve.

Now sometimes you will have signals that are slices of bigger signals that can also be concatenated so let's say you have a larger signal like a memory address

```vhdl
signal memory_address : std_logic_vector(15 downto 0);
signal lower_address : std_logic_vector(7 downto 0);
signal upper_address : std_logic_vector(7 downto 0);
```

You might want to split it into two like this

```vhdl
lower_address <= memory_address(7 downto 0);
upper_address <= memory_address(15 downto 8);
```

Then if you need to put them back you just concatenate

```vhdl
signal combined_address : std_logic_vector(15 downto 0);
combined_address <= upper_address & lower_address;
```

That's just the basics. Concatenation is super flexible in VHDL. You can concatenate literals too like this

```vhdl
signal my_data : std_logic_vector(15 downto 0);
my_data <= "0101" & x"A5";
```

That would make `my_data` equal to `010110100101` It's important to remember in vhdl the literal values are seen as strings and this is not an error. It makes for some good gotchas in the beginning of your VHDL learning journey. You need to be careful what you assign to what type of signal it can be a bit confusing the types in VHDL.

Now for some advanced stuff remember that image pipeline I was talking about Earlier I moved to a more sophisticated hardware architecture a few years later and started to use a frame buffer That required concatenating variable length data based on dynamically calculated offsets and lengths now that’s where things got interesting. I even used for loops and unconstrained arrays for that so I could change resolutions in a more flexible way I ended up using a loop to select chunks from the original data signal and then concatenate each chunk to output signal

And yes I had to debug that code too it took a while mostly because I kept using the same signal names in the loops that caused some serious head scratching

In short there’s a lot to watch out for here when writing code in VHDL especially the signal declarations and the types assigned to them so make sure you are consistent when it comes to this.

Also if you are new I'd recommend not going to the advanced stuff right away just get used to simple concatenations first and learn about vhdl types.

Oh and a little joke I heard once a boolean walks into a bar the bartender asks him what he wants he says "I'll take an or" you know because it can be either a true or false haha ah ah back to code.

And I’ve had my share of mistakes believe me including those fun ones where you forget you're concatenating and think you're doing math and you get some really weird looking outputs you think your logic is wrong but its just a concatenation issue so take it slow.

For resources you should really check out "Digital System Design with VHDL" by Zwolinski it’s a classic. Also try to read through the IEEE VHDL standards documents if you can it gets a bit dry but you will understand everything. If you need a more focused resource on VHDL synthesis and RTL design then look into "FPGA Prototyping by VHDL Examples" by Pong P Chu.

One more thing that sometimes people mix up you can use concatenation inside array assignments but it’s not always what you think you can have vector arrays and you can assign the whole array with concatenated vectors or specific positions so this is also a good area to watch out when dealing with concatenation.

And remember you can always use the `others` keyword inside concatenations sometimes if you are dealing with larger arrays so you don’t have to copy the whole thing. So something like `my_new_signal <= my_old_signal & (others => '0');` to pad the old signal with zeros. There are lot of things you can do with it.

Anyway that's about it for concatenation its fairly simple once you wrap your head around it but its also quite important in VHDL I hope that helps and if you have any other questions feel free to ask.
