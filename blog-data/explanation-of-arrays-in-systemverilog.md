---
title: "explanation of arrays in systemverilog?"
date: "2024-12-13"
id: "explanation-of-arrays-in-systemverilog"
---

Alright so you wanna talk about arrays in SystemVerilog right Been there done that got the scars to prove it I swear those multi-dimensional packed arrays gave me nightmares back in the day Let me break it down for you like I would for a newbie just starting out except you're not a newbie because you asked so lets go

SystemVerilog arrays are like containers for storing data They can hold variables of the same type think of it like a toolbox each compartment holding screws or maybe bolts but not both at the same time You got your basic types like int logic bit reg these are your building blocks then you can group them up with arrays into a larger structure

The first distinction you gotta get is between packed and unpacked arrays Packed arrays are where the data elements are crammed right next to each other in memory think of it like your Lego bricks all stacked together in a single column or maybe a flat brick shape and it forms a solid block Unpacked arrays are looser they can have gaps in memory and they’re more like individual storage bins You access packed arrays bit-wise like you would with a bus while unpacked arrays you access using indices meaning numbers to get at the element you need

Let’s say you’re working with a simple register you need to store 8 bits You can declare it using a packed array

```systemverilog
logic [7:0] my_register; // 8-bit register using a packed array
```

This creates a register named my_register that holds 8 bits of logic The [7:0] defines the size of the packed array and the direction it increments or decrements this case 7 to 0 so it is decreasing If you want to access bits you can use the [ ] operator like `my_register[3]` will get you the fourth bit counting from the right at 0 position Now if you want to assign a value to the whole thing or parts of it that's easy too You can use `my_register = 8'hAA;` which sets all the bits to `10101010` in binary that is the `AA` hexadecimal value or lets say for example that we wanted to set bit 3 to 1 only without changing anything else we can do `my_register[3] = 1'b1;` We can also declare a large set of them

```systemverilog
logic [31:0] data_bus [0:7]; // 8 32-bit data buses
```

Here data_bus is an unpacked array of 8 elements each of which is a 32-bit packed array. So think like 8 buses on a chip each 32 bits wide Now accessing them is a bit different First we access the bus number like `data_bus[2]` this gets us the third 32-bit bus then we can access a bit within that bus like `data_bus[2][15]` this gets the 16th bit of the third bus You can have more dimensions if you want

```systemverilog
logic [7:0] memory [0:255][0:15]; // 256x16 byte memory array
```

Now memory is a two dimensional unpacked array with 256 rows and 16 columns which are each 8 bit wide so it looks like some kind of memory matrix now access would be for example like memory[5][10] which will get you the byte at row 5 and column 10 but the address for each is an integer Remember that the dimensions for packed arrays are always given in [] before the name of the variable and unpacked arrays are after that so that is a very important part of the syntax that people often get wrong I remember when I first started I spent a whole weekend debugging just to see I had put the [ ] in the wrong place

Now here's the thing with multi-dimensional packed arrays they can get complicated quickly which I also learned the hard way like when we were simulating some complex DMA engine controller and we had tons of address buses with weird dimensions to emulate memory regions that were scattered around the silicon It was like trying to untangle a Christmas tree lights string after the party was over and the cat had gotten into it You wouldn't believe the time i spent going over waveform to track down a single bit flip

You can assign packed arrays just like regular variables but you gotta be careful with the size mismatches SystemVerilog isn't as forgiving as say Python or even C you can't just throw data in and hope it works So if you are assigning a smaller value to a large packed array it will automatically pad with zeros If you assign a larger value it will just truncate and give you a warning or even an error if you are doing strict checks which is good practice But doing strict checks can be a pain too you just gotta find a sweet spot

Unpacked arrays have their own quirks They're great for storing things like lookup tables or memory structures where you need to access individual elements without having to do bit manipulation or bit masking with a large number of bits you know you have to access each byte separately You can initialize them using the '{ }' syntax which is pretty convenient It allows you to assign the values in the declaration itself like `logic [7:0] my_table [0:3] = '{ 8'h11, 8'h22, 8'h33, 8'h44 };` this will initialize `my_table[0]` to `11`, `my_table[1]` to `22`, and so on which saves lots of code and makes it more readable It's better than assigning each of the addresses of the table like `my_table[0] = 8'h11; my_table[1] = 8'h22;` and so on Also remember that if you don't initialize them they will just start with `x` which is unknown and that will haunt you later so don't forget to initialize them

Another thing you should know is that you can also have dynamic arrays in SystemVerilog which don't have a fixed size but you need to allocate space for them at runtime This can be useful when you don't know the size of your data structures in advance for example if you are reading data from a file with variable amounts of data you would need a dynamic array They use the 'new' operator to allocate memory and you can use array methods like `size()` `push_back()` and `pop_back()` to manipulate them

Here’s a snippet showing the use of a dynamic array

```systemverilog
int dyn_arr []; // Declare a dynamic array of integers
initial begin
  dyn_arr = new[10]; // Allocate space for 10 integers
  foreach(dyn_arr[i]) dyn_arr[i] = i; // Initialize them to its index value
  dyn_arr.push_back(10); // Add an integer at the end
  $display("Size of the dynamic array %0d", dyn_arr.size()); // Size 11
  $display("Last element of the dynamic array %0d", dyn_arr[dyn_arr.size()-1]); // Get the last element
end
```

Using dynamic arrays adds an extra layer of flexibility but also complexity you gotta handle the allocation and dealocation of memory carefully to avoid memory leaks that are very hard to track you see something weird is happening but can not see it in the signals because it's in the memory I personally prefer statically allocated arrays most of the time It's the kind of situation where you have to think hard before making your choices

For learning more about this I’d recommend checking out “SystemVerilog for Verification” by Chris Spear It's like the bible of SystemVerilog and it covers all the details about arrays with examples and also the "Verification Methodology Manual for SystemVerilog" also known as the VMM it covers the array usage in a verification setting both are fantastic resources and every SV engineer should have these in their personal library Another great thing is the online IEEE standard 1800-2017 if you want a reference that's the one to use but it can be a bit dense to read in one go

So that's the gist of it Arrays are fundamental to working with hardware in SystemVerilog and you will use them all the time It takes some time to wrap your head around all the different types but once you do it gets much easier Trust me I've spent countless hours debugging arrays and believe me some of those hours could have gone to better use for example I could have started working early on my next side project I have ideas you wouldn't believe it's about using a high speed interface to control a laser display but I keep getting delayed from the current project It's all arrays everywhere

Anyways the important thing to remember is that SystemVerilog is a very versatile language and it can take you a long way but mastering the basics first is extremely important that's why I told you everything I know about the subject so I hope that I helped you I spent a lot of my time answering this which is what stackoverflow is about so I'm happy I could help
