---
title: "casting packed array to unpacked array for arrays that are used as parameters?"
date: "2024-12-13"
id: "casting-packed-array-to-unpacked-array-for-arrays-that-are-used-as-parameters"
---

Okay so you want to cast a packed array to an unpacked array when those arrays are function parameters right I've been down that rabbit hole more times than I'd like to admit Let me share some scars and hopefully save you some headaches

First off the packed vs unpacked thing in hardware description languages like SystemVerilog can be a real pain especially when it comes to interfaces and function calls You see when you define an array as packed it means all the bits are stored contiguously in memory as one large chunk Think of it like a single long register Unpacked on the other hand is treated more like a collection of individual registers or memory locations Each element exists in its own little space

Now when you try to pass a packed array as an unpacked parameter or vice versa you get this type mismatch error It's the compiler saying "hey these are not the same" And it's right

The easiest solution is often to manually unpack the packed array inside the function This means creating a new unpacked array within the function and copying each element from the packed array This is not the most elegant way but it's clear straightforward and avoids most casting problems You'll need to know the sizes for both arrays

```systemverilog
function void my_function (input bit [31:0] packed_array input bit [7:0] unpacked_array_in [3:0] );
  bit [7:0] unpacked_array [3:0];
  for (int i = 0; i < 4; i++)
      unpacked_array[i] = packed_array[(i*8)+:8];
   $display("unpacked array: %p",unpacked_array);

  // Do something with unpacked_array
  //... some code...
   for (int i = 0; i < 4; i++)
      unpacked_array_in[i] = unpacked_array[i];

endfunction

```

In this example I create an unpacked array *unpacked_array* inside the function that's the same size as needed then loop through and copy over each 8 bit chunk from the *packed_array* You can adjust the loop and indexing to match the width and depth of the arrays in your problem. I also added the *unpacked_array_in* so you can modify that one on the function as part of the parameters. Remember that this is done inside the function scope

This works but is inefficient Especially when you have multiple functions and the copying overhead is killing performance and takes more code. Now you might think why not just use a direct cast like this:

```systemverilog
function void my_other_function(input bit [31:0] packed_array input bit [7:0] unpacked_array_in [3:0]);
  bit [7:0] unpacked_array [3:0];
  unpacked_array = unpacked_array'(packed_array);
  $display("unpacked array: %p",unpacked_array);
   for (int i = 0; i < 4; i++)
      unpacked_array_in[i] = unpacked_array[i];
  // do other things
endfunction
```

This example tries a casting that doesn't really works the casting operator *'()* performs a conversion that might not always translate from a packed format to an unpacked format. This could cause a mismatch and wrong data. Also this cast does not respect the width of each chunk it's rather a memory interpretation cast that can create unwanted effects

The real magic here is using a properly structured *typedef* This approach takes more setup but pays off in the long run You create a user defined type that has both the packed and unpacked versions of your array Then the function just uses the user defined type as an argument

```systemverilog
typedef struct packed {
    bit [31:0] packed_data;
    } packed_type;

typedef struct {
    bit [7:0] unpacked_data [3:0];
} unpacked_type;

function void my_last_function(input packed_type data_in, output unpacked_type data_out);
   bit [7:0] unpacked_array [3:0];
  for (int i = 0; i < 4; i++)
      unpacked_array[i] = data_in.packed_data[(i*8)+:8];
  data_out.unpacked_data = unpacked_array;

   $display("unpacked array: %p",data_out.unpacked_data);
endfunction
```

This might look a bit more complicated but it is worth it The *typedef* makes code a lot more readable and also safer The key here is to think about the data and how its accessed in your design. I personally prefer this last approach because it gives you more control and readability

Now you have this *packed\_type* and *unpacked\_type* So when you have a function that needs an unpacked version of your data you declare a struct with it When you need the packed version you just pass the struct I know it might seem like a detour but trust me it reduces errors and makes your code much much easier to work with down the road

There is this older paper called "SystemVerilog for verification" by Chris Spear. Its a bit old but the fundamentals of typedefs are explained there Also there is another book called "Verification Methodology Manual for SystemVerilog" from Janick Bergeron that it is very good to better understand the different uses of user defined types and how parameters should be created. It also covers in detail all the verification approaches in systemverilog so is a good all around book for the language. It's a worth read if you are serious about hardware design and verification

Remember that these are not some theoretical computer science ideas These are real world problems I've spent hours debugging so I learned these the hard way at one point I had a module that was sending data to a core that expected an unpacked array it was a late Friday evening debug session I ended up calling it a week and came to the office on Saturday to find out that a stupid cast caused everything to misbehave the lesson was clear never trust a simple typecast without understanding the memory layout involved it was so bad that I almost started using VHDL I mean who would do that Seriously haha.

So to summarize:

1.  **Manual unpacking:** Simple and clear but less efficient for a lot of data operations
2.  **Type casting:** Can work but often leads to errors and unexpected results especially with packed types
3.  **User defined types (typedef):** Most robust and maintainable but it needs a little more setup

So think carefully about your design and try to avoid mixing the packed and unpacked worlds as much as you can. The best solution depends on your specific case but I'd recommend the user defined types for any more complex designs where performance is key or safety and stability are needed. Don't make my mistake and learn the hard way by spending nights awake trying to figure out a problem created by something you thought it would work.
