---
title: "case statement in verilog?"
date: "2024-12-13"
id: "case-statement-in-verilog"
---

Alright let's dive into this verilog case statement thing I've wrestled with this beast more times than I care to remember and honestly its still a point of minor frustration even after all these years So you wanna know about case statements in verilog right Cool beans let's break it down in a way that even my past self struggling with timing closure would understand

First things first a case statement in verilog is essentially a multi-way branch It's your go-to when you have a single expression and you want different things to happen based on the value of that expression Think of it like a giant if-else chain but way more elegant and readable at least once you get the hang of the syntax

Now for the nitty-gritty the basic structure is something like this

```verilog
case (expression)
  value1: statement1;
  value2: statement2;
  value3: statement3;
  default: default_statement;
endcase
```

The `expression` is what you're evaluating It can be a single signal or a combination of signals and the `value` items are what you are comparing it against If the `expression` matches the `value` the corresponding `statement` gets executed If nothing matches the `default` statement gets executed and if there is no `default` statement and none match then nothing happens just like a silent fail which we all know is the worst

Okay so that's the most basic form lets get a bit more concrete and give you a code snippet to actually make sense of it because we're engineers not poets or whatever right? This is what we used in our first project which got absolutely rekt by the test bench because of an issue with the reset logic we had to re-write the entire thing from scratch so yeah this thing gives me PTSD

```verilog
module simple_case (
  input [2:0] state_input,
  output reg [3:0] output_val
);

  always @(*) begin
    case (state_input)
      3'b000: output_val = 4'b0001;
      3'b001: output_val = 4'b0010;
      3'b010: output_val = 4'b0100;
      3'b011: output_val = 4'b1000;
      default: output_val = 4'b0000;
    endcase
  end

endmodule
```
So what's happening here? We've got a module named `simple_case` with a 3-bit input called `state_input` and a 4-bit output register called `output_val` The `always @(*)` block means this logic will be re-evaluated anytime any of the inputs in the sensitivity list changes which is basically everything here The `case` statement then checks the value of `state_input` and if its `000` `output_val` becomes `0001` if its `001` `output_val` becomes `0010` and so on If `state_input` is any other value other than the ones listed the `default` statement sets it to `0000`

A couple of things to notice here Firstly verilog is picky about width mismatches so you need to make sure your `value` widths match the width of your `expression` Second the `default` statement is optional but its generally a good idea to have one It can prevent unexpected behavior in cases you haven't explicitly covered with another case statement

Now some people run into the `casez` and `casex` statements which are kinda like special case scenarios and there's a difference between them `casez` treats any z (high impedance) in a value as a don't care `casex` treats both z and x (unknown) as don't cares They are very useful in test benches or for situations where you don't care about parts of your expression but be careful using them in actual RTL because it may result in unwanted behavior if you don't understand how the synthesizer interprets it

Let me give you a practical example I had once where I had to implement a 7 segment display controller and I used `casez` to display the numbers from 0 to 9 This was during my internship after my disastrous first attempt which I swear I barely passed It was a nightmare trying to debug it at 3 am with a caffeine overdose
```verilog
module seven_segment_display (
  input [3:0] digit,
  output reg [6:0] segments
);

  always @(*) begin
    casez (digit)
      4'b0000: segments = 7'b1000000; // 0
      4'b0001: segments = 7'b1111001; // 1
      4'b0010: segments = 7'b0100100; // 2
      4'b0011: segments = 7'b0110000; // 3
      4'b0100: segments = 7'b0011001; // 4
      4'b0101: segments = 7'b0010010; // 5
      4'b0110: segments = 7'b0000010; // 6
      4'b0111: segments = 7'b1111000; // 7
      4'b1000: segments = 7'b0000000; // 8
      4'b1001: segments = 7'b0010000; // 9
      default: segments = 7'b1111111; // Turn all off on invalid inputs

    endcase
  end

endmodule
```

Okay that's a bit more involved In this one our `digit` input is now 4-bits and it controls which digit is displayed on our seven segment display The `casez` is actually acting as a normal case here because we don't have any z values It's simply to show that if we had used dont cares they would have been taken into account. The `segments` is a 7-bit output that drives the display segments if we want to display a `2` for example we'd set the `segments` to 0100100 by decoding it from the input `digit` using the casez statement.

One thing that can cause issues sometimes and trust me I've been there is when you have multiple matches in your `case` statements Now Verilog will only execute the very first match so make sure to pay attention to the order of your `value` items

Also don't try to be too clever with the case statement logic because that can get you into trouble faster than you can say timing closure I've seen some complex cases that end up creating huge complex and inefficient hardware that takes hours to compile and that no one really understands If someone cannot understand the logic after one single look you messed up hard

Let me show you one of the examples of a code that was too complex I was involved with This code was such an absolute disaster that I still cringe thinking about it

```verilog
module bad_case (
  input [3:0] input_bus,
  output reg [3:0] output_bus
);

  always @(*) begin
    case (input_bus)
      4'b0000: output_bus = 4'b0001;
      4'b0001: output_bus = 4'b0010;
      4'b000x: output_bus = 4'b0011;
      4'b0010: output_bus = 4'b0100;
      4'b0011: output_bus = 4'b0101;
      4'bx000: output_bus = 4'b0110;
      4'b1000: output_bus = 4'b0111;
      default: output_bus = 4'b0000;
    endcase
  end
endmodule
```
You see the problem right We are using the dont cares (x) but that logic is not correctly implemented in the case statement because since the first matches are `0000` and `0001` then `000x` will never be reached and will simply be skipped because the first one matches it. This will cause unwanted behavior because we think it will match our x case when in reality the earlier cases are matching.

So yeah that's the gist of case statements in verilog I know it can be a bit daunting at first but once you understand the fundamentals and do some practice you'll be a case statement master in no time just remember that less complexity is always better and if you find yourself writing complex logic that you yourself can barely understand its better to re-write it and that's what I had to learn the hard way back in the day. Also don't forget that synthesis tools can sometimes surprise you and it's always better to double check that what you wrote matches the expected behavior. Oh and this is probably obvious but make sure your test benches are complete because I remember one time I spent hours trying to figure out a bug only to find out that I never even tested a specific edge case which was the real problem.

Now if you're looking for more in-depth knowledge I suggest reading through "Digital Design: Principles and Practices" by John F Wakerly or "Computer Organization and Design" by David Patterson and John Hennessy These books are like the bible for digital design and will give you a solid foundation Also there are numerous online tutorials and courses that can help you with specific aspects of verilog.
Remember the journey of a thousand lines of code starts with a single line or in our case a single case statement so don't get discouraged and keep coding.
