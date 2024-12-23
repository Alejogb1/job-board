---
title: "in system verilog and or operators?"
date: "2024-12-13"
id: "in-system-verilog-and-or-operators"
---

 so you're asking about `and` and `or` operators in SystemVerilog yeah I've wrestled with those little critters more than I care to admit let me tell you

First things first you need to understand there are actually two kinds of these logical operators in SystemVerilog the bitwise versions and the logical versions they seem the same but they act on different things and produce different results it's a common gotcha so pay close attention

 so let's dive right in with bitwise operators which are denoted by a single `&` for bitwise and and a single `|` for bitwise or I’m old enough to have coded a flip flop using only NAND gates by hand let me tell you those were the days and the experience is invaluable trust me these guys operate bit by bit on the operands they require each operand to be a multi-bit value and the operation happens on each corresponding bit position of both operands

Like if you have something like this

```systemverilog
logic [3:0] a = 4'b1010;
logic [3:0] b = 4'b1100;
logic [3:0] result_and;
logic [3:0] result_or;

assign result_and = a & b; // result_and will become 4'b1000
assign result_or = a | b;  // result_or will become 4'b1110
```

See how the bits match up? it's bit 0 of `a` with bit 0 of `b` bit 1 of `a` with bit 1 of `b` and so on the bitwise `and` spits out a `1` only if both bits are `1` otherwise it's `0` on the other hand bitwise `or` gives you a `1` if at least one of the bits is `1`

Now this is critical don’t try to use these on single-bit values like boolean flags unless that's what you actually want as you’ll get unexpected outcomes a few years back i was debugging this complicated state machine with multiple complex flags and my flags behaved weirdly it took me two days of deep diving and I found out that I had mixed logical and bitwise operators in the control logic that is a no no and that's when I understood the difference fully

 so what happens if operands have different bit widths well system verilog takes the smaller width and expands it by padding on the left with zeros this is called zero extension so if you mix for example

```systemverilog
logic [7:0] long_data = 8'hAA; // 10101010
logic [3:0] short_data = 4'h5; // 0101
logic [7:0] result_or_diff_size;

assign result_or_diff_size = long_data | short_data; // result_or_diff_size is 8'hAF;  // 10101111
```

The short_data `4'b0101` becomes `8'b00000101`  then you get the bitwise `or` operation then the result is `8'hAF` which is `10101111` in binary

Now let's switch to the logical operators these are `&&` for logical and and `||` for logical or these treat their operands as boolean values where zero means false and anything that is non-zero means true they return a single bit result `1` for true `0` for false

So something like this

```systemverilog
logic a_bool = 1;
logic b_bool = 0;
logic result_and_bool;
logic result_or_bool;

assign result_and_bool = a_bool && b_bool; // result_and_bool is 0 because b_bool is 0
assign result_or_bool = a_bool || b_bool;  // result_or_bool is 1 because a_bool is 1
```

Makes sense right? Even if `a_bool` and `b_bool` where multi-bit signals the logic would look at them as true or false not bit-by-bit. So if `a_bool` was `4'b0100` then the expression `a_bool && b_bool` would return zero because `b_bool` is zero no matter the bits in `a_bool` because as long as there is at least one bit which is set it will be true and in the case of all bits being zero the signal will be treated as false.

It's really a big thing not to confuse logical and bitwise operations because of their apparent simplicity especially if you are working with complex multi-bit signals like buses in a processor or a custom chip this can take a lot of time debugging if you mess them up trust me I’ve been there done that even got the T-shirt! haha ok sorry about that

Also there is the short circuiting feature which is very important in logical operations. In an `and` operation if the first operand evaluates to zero SystemVerilog doesn't even bother checking the second one since the final answer is going to be false no matter what in `or` operation if the first is one then it doesn't check the second one as the result will be one also this has big impacts on simulation performance and how your expressions get evaluated and this is where things get tricky especially if you try to embed function calls in logical operations. You might find out that your function is not being called when you expect it because of this short circuiting feature be aware of it and the order of the expressions is very important in these situations I've had to reorder a few expressions in my code after being bitten by this so pay attention

Here is another example of short-circuiting in action

```systemverilog
logic condition1 = 0;
logic condition2;
integer result;

function integer some_function();
   $display("some_function called");
   return 1;
endfunction

initial begin
    result = condition1 && some_function();
    $display("result is %d", result);
    result = 1 || some_function();
    $display("result is %d", result);
    result = 1 && some_function();
    $display("result is %d", result);
end
```

in the code above the function is called only one time and only in the last case because condition1 is 0 and it short circuits the second operand in the first operation and also for the second operation with the "or" operator

So to summarize bitwise operates bit by bit logical operations deal with truth or false this makes them suitable for different kinds of things like bit masks or boolean flags

For good resources to understand this in detail you might want to check "SystemVerilog for Verification" by Chris Spear it's a classic and its very useful for these details. Also the IEEE 1800-2017 standard for SystemVerilog itself is very exhaustive.
