---
title: "arithmetic shift in verilog signed?"
date: "2024-12-13"
id: "arithmetic-shift-in-verilog-signed"
---

so you're wrestling with arithmetic shifts in Verilog specifically when dealing with signed numbers I feel you I’ve been there trust me It’s one of those things that seems simple on the surface but can bite you in the butt if you don't understand the nitty-gritty

I remember back when I was working on a custom DSP block for some image processing thing I needed to do some seriously fast and precise calculations and yeah you guessed it I messed up my signed shifts and the whole system started spewing out garbage results It took me a good solid day of debugging to figure it out it was a real head scratcher let me tell you

Here’s the deal with arithmetic shifts and signed numbers in Verilog The crucial thing to remember is that arithmetic shifts preserve the sign of the number unlike logical shifts This matters a whole lot when working with signed data because the most significant bit the leftmost bit encodes the sign if its one the number is negative if its zero its positive You don't want your shifts to mess with that

A logical shift just moves all bits to the right or left regardless of what is in the MSB if you right-shift you insert zeros from the left and vice versa if you are doing a logical left-shift Arithmetic shift on the other hand treats the number as signed during right shifts if you right-shift the MSB is copied to the right this is called sign extension and that’s what preserves the sign of a negative number

Now the Verilog syntax for it its pretty straightforward you use the `>>>` operator for an arithmetic right shift and the `<<` operator for an arithmetic left shift Yes both operators are the same as logical shifts but if the operand is defined as signed it will do arithmetic shifts if the operand is unsigned it will do a logical shift

Lets dive into some actual code examples

**Example 1 Arithmetic Right Shift**

```verilog
module arith_shift_example;

  reg signed [7:0] signed_val;
  reg signed [7:0] shifted_val;

  initial begin
    // Positive number shift
    signed_val = 8'sd16; // Decimal 16
    shifted_val = signed_val >>> 2; // Right shift by 2
    $display("Original positive value: %d  Shifted value: %d", signed_val, shifted_val); // Expected 4

    // Negative number shift
    signed_val = 8'sd-16; // Decimal -16
    shifted_val = signed_val >>> 2; // Right shift by 2
    $display("Original negative value: %d Shifted value: %d", signed_val, shifted_val);  // Expected -4

    // Another negative shift
    signed_val = 8'sd-1; //Decimal -1
    shifted_val = signed_val >>> 3;
    $display("Original negative value -1 Shifted value: %d", shifted_val); //Expected -1

  end

endmodule

```

In this example you can see the sign bit is preserved during the right shift the `>>>` operator. The positive number is shifted as expected and the negative number retains it's negative sign thanks to sign extension with that MSB copying when we shift right.

**Example 2 Arithmetic Left Shift**

```verilog
module arith_left_shift;

  reg signed [7:0] signed_val;
  reg signed [7:0] shifted_val;

  initial begin
    // Left shift and multiplication
    signed_val = 8'sd3; // Decimal 3
    shifted_val = signed_val << 2; // Left shift by 2 (effectively multiplies by 4)
    $display("Original value: %d Left shifted value: %d", signed_val, shifted_val); //Expected 12

    //Left shift negative number
    signed_val = 8'sd-2;
    shifted_val = signed_val << 2;
    $display("Original negative value: %d Left shifted value: %d", signed_val,shifted_val); //Expected -8

    //Left shift overflowing
    signed_val = 8'sd60; // Decimal 60
    shifted_val = signed_val << 1;
    $display("Original positive value: %d Left shifted (overflowing) value: %d", signed_val, shifted_val); // Expected -112 due to overflow

    signed_val = 8'sd-60; // Decimal -60
    shifted_val = signed_val << 1;
    $display("Original negative value: %d Left shifted (overflowing) value: %d", signed_val, shifted_val); //Expected 40 due to overflow

  end
endmodule

```

Here we're demonstrating that the arithmetic left shift `<<` works as a simple multiplication by powers of 2 but you have to be careful because if you shift too far on the left you can have an overflow and the data you get is unreliable. It depends on what you want if you expect a modulus or not but the behavior is going to be an unexpected sign change for sure

**Example 3 Signed vs Unsigned Shift Comparison**

```verilog
module signed_unsigned_shift;

    reg [7:0] unsigned_val;
    reg signed [7:0] signed_val;
    reg [7:0] unsigned_shifted;
    reg signed [7:0] signed_shifted;

    initial begin
    //Unsigned value shifts
    unsigned_val = 8'd250; // Unsigned value 250
    unsigned_shifted = unsigned_val >> 2; // Logical right shift
    $display("Unsigned original value %d unsigned shifted value: %d", unsigned_val, unsigned_shifted); // Expect 62

    //Signed values shifts
    signed_val = 8'sd-6; // Signed value -6
    signed_shifted = signed_val >>> 2; //Arithmetic right shift
    $display("Signed original value %d signed shifted value: %d", signed_val, signed_shifted); //Expect -2

    //Same bit pattern on unsigned but shifted logically
    unsigned_val = 8'd250;
    unsigned_shifted = unsigned_val >>> 2;
    $display("Unsigned bit pattern shifted logically :%d result: %d", unsigned_val, unsigned_shifted); //Expect 62

    end
endmodule

```
This example shows the differences between the two kind of shifts when you have signed and unsigned numbers doing an arithmetic or logical shift

Now one of the things I had trouble with early on is I forgot the signed and unsigned definitions of data which results in incorrect results as the hardware synthesizers default the data type to unsigned if you do not specify and you get the logical shift instead of the arithmetic one even if you expect one This can lead to very difficult to debug errors that you will never find I always use signed type to make my data type and intent clear especially if dealing with signals that need a sign

Another thing you should always keep in mind is that you can only use shifts using constant values which you have already defined in your design and not by an input signal that is something you can not do unless you have something like barrel shifter where your shift quantity is input data but that is another story and a more complex design and you will require extra hardware

Now the question you should be asking is when do we use a right arithmetic shift well it’s particularly useful when you are dividing by powers of two. And of course with negative numbers you need the sign extension to get the expected results and not just have an unsigned right shift that will mess your numbers.

Left shift its usually used to do multiplications by powers of 2 and sometimes as part of more complex operations where you combine them with addition or subtraction its essential to make sure your bit width is enough so you can handle the results of those operations without overflow

If you really want to dig deeper and understand these topics I would recommend not looking for online resources that might be questionable instead I would suggest some good old classic books and papers start with "Digital Design Principles and Practices" by John F. Wakerly its a solid resource that gives you the basics of digital design including these shift operations. Another resource I used in my school days was "Computer Organization and Design" by Patterson and Hennessy although it's not specifically focused on Verilog it provides a fantastic background in how these operations work on actual hardware that will give you some more context. There are also some IEEE papers that might contain some specific edge cases that might be useful but they are pretty complex and not required to know

And one last joke for you why did the integer break up with the float? Because they had too many decimal points in their relationship ha I tried

So yeah that's the lowdown on arithmetic shifts with signed numbers in Verilog it's about understanding sign extension and remembering to define your signals as signed. Always test your code with both positive and negative values to make sure everything behaves as expected Happy coding and remember to always check those overflow bits !
