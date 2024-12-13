---
title: "multiple items matching in verilog case statement?"
date: "2024-12-13"
id: "multiple-items-matching-in-verilog-case-statement"
---

Okay so you’re asking about multiple items matching in a Verilog case statement huh classic problem I've been there done that bought the t-shirt and probably even the coffee mug

Let me tell you it's one of those things that beginners stumble on all the time and even experienced folks sometimes get a little tripped up on because it’s not as straightforward as you might first think you’ve got a bunch of conditions and you want the same action for several of them but you're not entirely sure how to express that cleanly in Verilog

Okay let's dive in to how this thing works and some of the nuances of the language

First things first what's the deal with a Verilog case statement

You’ve probably used a switch statement in a higher-level language it’s similar but with a Verilog twist you have your case expression which is some input signal or variable and then you have a bunch of case items these case items are the values you’re comparing your case expression against if a match is found the block of code associated with that item is executed

So where does the "multiple matching" part come in I'm betting you've probably tried something like this

```verilog
case (my_input)
  1: my_output = 1;
  2: my_output = 1;
  3: my_output = 2;
  4: my_output = 2;
  default: my_output = 0;
endcase
```

Right this looks like it should work it looks like you are trying to set my_output to 1 for both case 1 and 2 and to 2 for both cases 3 and 4  But Verilog doesn't work quite like that it'll match the first case and execute it and then that’s it the execution will not continue down the list of the case items

Here's the catch Verilog case statements don't behave like a C or Java switch statement they execute at most a single case they aren't meant to go looking for multiple matching case values

So how do we handle multiple matching items in Verilog

Well there are several common ways

The most direct way is to just list out each case value separately and use the same block of code for each case as such

```verilog
case (my_input)
  1: my_output = 1;
  2: my_output = 1;
  3: my_output = 2;
  4: my_output = 2;
  default: my_output = 0;
endcase
```

This will work as intended as the execution will stop after the case statement is evaluated for the single matching item but if you want a better version that looks more concise you can use the following which is the most common pattern

**Use the comma separated case list**

This is the most typical way to handle it in the verilog world You simply list the case values separated by commas like so

```verilog
case (my_input)
  1, 2: my_output = 1;
  3, 4: my_output = 2;
  default: my_output = 0;
endcase
```

This is equivalent to the previous example but it’s more compact and easier to read When any one of those values 1 and 2 are matched then the value of 1 is assigned to my_output This method gets a little unruly if you've got a long list of items that should have the same action but is is the most used one I have seen in the real world

Now when would I use one over the other

I usually use the comma method but if I want to make it extremely explicit or if I need specific comments next to a particular case then I will use the separate case statements as shown in the first example which is something that can happen when the person writing the code is not the person that originally conceived it

**Range Matching**

Now here's where it gets slightly more interesting what if you have a contiguous range of items instead of just a few scattered values

Verilog lets you specify a range using the colon operator like this

```verilog
case (my_input)
  2'b00: my_output = 1;
  2'b01: my_output = 1;
  2'b10, 2'b11: my_output = 2;
  default: my_output = 0;
endcase
```

this would work similarly to the last example but you can represent a range more explicitly something like `4:7: my_output = 3;` in this case it would match 4 5 6 and 7 and make the assignment

Note that range matching also works with named parameters so for example if you had `parameter START = 4; parameter END = 7;` you can use `START : END: my_output = 3;` which can be pretty handy if you want to move this code to another design

The thing I find funny is that after all these years of using hardware languages I still tend to mess up the `casez` statement when I am trying to do don't care matching in my case statements it is like the language is actively trying to mess me up somehow

**Case statements with don't care values**

Sometimes you have situations where you just don’t care about the specific value of some of the bits of the input signal In that case you can use `casez` to represent the don't care cases `?` or `z` to indicate those don't care conditions

```verilog
casez (my_input)
  4'b10?? : my_output = 1; //Matches 1000 1001 1010 1011
  4'b01z0: my_output = 2; //Matches 0100 and 0110
  default: my_output = 0;
endcase
```

In this example the `?` and `z` acts as a wildcard letting the `casez` statement match multiple values based on the ones that are set

Now I’ve used `casez` and don't care matching in countless different ways over the years but I think one of the strangest cases involved a controller for a memory subsystem. The input to that controller had a bunch of bits but many combinations of the input bits did not matter so it was much easier to specify what not to do using don’t care bits

I remember debugging that one for a few hours thinking that my logic was faulty only to find out that I had a small typo in my `casez` statement. I was using `10??` instead of `10z?` and that caused some really weird behavior

**Other stuff to watch out for**

One other thing to be aware of is that the default case is really important. You should always have a default case in the majority of situations in your `case` statement otherwise you may create unintended latches because if you don’t specify what the output is when none of the cases match the synthesis tool might not generate anything and you will get unpredictable results which in my experience will be a nightmare to track

Also think about your encoding whenever you are writing case statements If you are doing comparisons with enumerated types use the type itself rather than just an integer literal this will help you when writing and debugging your code for example `state = IDLE;` instead of `state = 0;`

In terms of resources if you’re diving deeper into Verilog I’d highly recommend you grab a copy of “Digital Design Principles and Practices” by John Wakerly or “Verilog HDL” by Samir Palnitkar both of them are extremely useful resources to learn not only Verilog but a bunch of practical things about writing hardware designs

You should also look into the IEEE 1364 standard as that’s the official reference for Verilog but it’s not exactly light reading so I’d say stick to the book approach if you are learning the language

So there you have it multiple item matching in Verilog is not difficult but it is something that you should be careful about it's one of those small things that if you understand and do well can save you a ton of time debugging down the line
