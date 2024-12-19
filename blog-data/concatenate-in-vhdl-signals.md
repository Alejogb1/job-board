---
title: "concatenate in vhdl signals?"
date: "2024-12-13"
id: "concatenate-in-vhdl-signals"
---

Alright so you're asking about concatenating signals in VHDL right Been there done that got the t-shirt And a few gray hairs from debugging those pesky signal assignments trust me I've seen it all the good the bad and the downright weird when it comes to VHDL concatenation

First things first let's talk about why you'd even want to do this concatenating signals is fundamental for building larger data structures from smaller ones imagine you have say a 4 bit bus and a 2 bit bus and you want to treat them as a single 6 bit bus for some reason that's where concatenation comes in handy it's like assembling lego blocks into a bigger more complex structure that's kind of the basic principle

Now VHDL offers this nifty operator & for concatenation it's not the same as bitwise and though be careful to not mix up those two it’s an important distinction and a very common beginner mistake I’ve seen my fair share of debugging sessions that could’ve been avoided if this was clear

The syntax is pretty straightforward you just use the & operator between the signals you want to concatenate the result is a signal that has the bits of the concatenated signals lined up in the order you specified For example `result <= signal1 & signal2;` will create a new signal that has bits of `signal1` on the higher bits and bits of `signal2` in the lower bits that's usually how it goes with this operator

Let’s move on to actual code examples because everyone knows that's the real meat of the deal

Example 1 lets keep it simple just two std\_logic\_vector signals I've been burned before by not declaring the types properly so let's be explicit about the std\_logic\_vector usage

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity concatenation_example_1 is
    Port ( signal_a : in  std_logic_vector (3 downto 0);
           signal_b : in  std_logic_vector (1 downto 0);
           result   : out std_logic_vector (5 downto 0)
           );
end entity concatenation_example_1;

architecture Behavioral of concatenation_example_1 is

begin
  result <= signal_a & signal_b;

end architecture Behavioral;
```

In this simple example `signal_a` is a 4 bit wide vector and `signal_b` is a 2 bit wide vector The concatenation `signal_a & signal_b` results in a 6 bit vector where the most significant 4 bits are from `signal_a` and the least significant 2 bits are from `signal_b` I’ve seen code where the vector ranges were confused and the bits were concatenated in reverse order that usually leads to very nasty and strange debugging issues

Example 2 this is a classic case lets concatenate a bunch of single bit signals to create a bus it's more complex this time but still easy enough to wrap our heads around

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity concatenation_example_2 is
    Port ( bit0   : in  std_logic;
           bit1   : in  std_logic;
           bit2   : in  std_logic;
           bit3   : in  std_logic;
           bit4   : in  std_logic;
           bus_out : out std_logic_vector (4 downto 0)
           );
end entity concatenation_example_2;

architecture Behavioral of concatenation_example_2 is

begin
    bus_out <= bit4 & bit3 & bit2 & bit1 & bit0;

end architecture Behavioral;
```

Here we are concatenating five single bit `std_logic` signals into a 5-bit `std_logic_vector` This is a super common pattern I actually used it once when I was implementing a simple data bus interface I had a register of individual control bits that I needed to combine into a single control word this kind of concatenation saved me a lot of headaches you can always be sure this kind of stuff will be needed somewhere in complex designs

Example 3 now this time we have to pay special attention here mixing signals and constants It’s super useful but can bite you if you are not careful with the types so that you won't get unexpected simulation results I’ve spent a couple of days debugging code with this precise problem

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity concatenation_example_3 is
    Port ( signal_c : in  std_logic_vector (2 downto 0);
           result   : out std_logic_vector (6 downto 0)
           );
end entity concatenation_example_3;

architecture Behavioral of concatenation_example_3 is

begin
  result <= signal_c & "101" & '0'; -- Careful with type matching
end architecture Behavioral;
```

In this example we concatenate a 3-bit `signal_c` with the string literal `"101"` which is interpreted as 3 bits and a single bit character literal '0' note the single quotes here its important This demonstrates how you can mix signals with constants as needed make sure you pay attention to the type matching here it’s the one thing that always brings problems to everyone It's one of the biggest gotchas in VHDL if you ask me

One thing to be very very very aware of is type consistency when you are concatenating signals make sure the types match up correctly You are not gonna be able to concatenate say a `std_logic` with an `integer` directly VHDL is strongly typed which is a blessing and a curse If you try to do it the compiler or simulator will most certainly complain about it Also very important make sure you understand the bit ordering of the vector you are creating if you flip the order you will have to deal with hours of debugging and probably very weird behavior as a result remember you have to get the order right

When it comes to troubleshooting it really is very simple in most cases if you see a synthesis or simulation error check the sizes of the vectors involved ensure the type match and that the sizes make sense a very small mistake can cause so many problems If you are concatenating a lot of stuff make sure to always double check and if possible triple check your sizes and type assignments remember the number of bits on each signal and how much of them you are using for the concatenation I once had a bug because I was off by one bit and spent a day debugging it that's VHDL for you

Regarding resources for more in-depth knowledge on VHDL concatenation and related topics I would suggest you to check out Ashenden's "The Designer's Guide to VHDL" it's a classic it's really a very deep dive into everything you would ever need and also there's "Digital Design Principles and Practices" by Wakerly this is a really good practical perspective into the matter and if you want something very very practical check out Pong P. Chu's book "FPGA Prototyping Using VHDL Examples" that book really saves the day when you want something practical

You know someone told me once that VHDL is like trying to build a house with instructions written in hieroglyphs but that's also why we like it I guess right Anyway I hope this helps I've been there many times with this stuff let me know if you got any other questions we all do at some point so lets be on the lookout to help each other out
