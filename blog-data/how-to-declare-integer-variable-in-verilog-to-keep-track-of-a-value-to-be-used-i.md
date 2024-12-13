---
title: "how to declare integer variable in verilog to keep track of a value to be used i?"
date: "2024-12-13"
id: "how-to-declare-integer-variable-in-verilog-to-keep-track-of-a-value-to-be-used-i"
---

Alright so you wanna declare an integer in Verilog to keep track of a value lets say for indexing right I've been there countless times and let me tell you it’s easy once you grasp the fundamentals it's not like rocket science or anything fancy just basic digital design stuff and verilog does the job very well

First things first you need to understand how verilog handles data types it's a strongly typed language that means you have to be explicit about what kind of data you are storing unlike some other languages we all know and love well some of us anyway for integers we use the `integer` keyword. Simple right

Now you might be thinking ok I declare it that's it nope there is a little bit of difference depending on where you need to declare it and use it we are talking about always blocks and modules

For example inside an always block you will declare it like this

```verilog
always @(posedge clk) begin
  integer my_index;
  my_index <= my_index + 1; // Example incrementing
  // Other logic using my_index
end
```

See how it's done inside begin end block simple easy to understand and we are in a always block we are using non blocking assignment the `<=` so that everything executes in parallel. No blocking assignment ` = ` is generally a big no no unless you know what you are doing and you are working with combinational logic which is not the case here

Also we can declare the same inside modules in a similar way as follows

```verilog
module my_module (input clk, output reg [7:0] data_out);
  integer my_counter;

  always @(posedge clk) begin
    my_counter <= my_counter + 1;
    data_out <= my_counter [7:0];
    end
endmodule
```

Again nothing complicated here the integer `my_counter` is declared inside the `module` but not inside any always block itself this makes `my_counter` available inside any `always` block inside that particular module

A few years back I was working on a pretty complex image processing unit that's when I really learned how important these integer indices are I was trying to process a very large 2D image frame by frame and I needed to keep track of both the row and column indices of the pixels I was working on initially I tried using a bunch of `reg` variables thinking they were the same but I ran into all sorts of weird issues with how synthesis tools were optimizing my design it took me some time and head scratching to figure out that I should have used integers instead of registers for the loop counters lesson learned. It’s very very important to keep in mind what you're doing and the use case otherwise the synthesized digital logic might act weirdly and give you very headache

Now let me show you something that may come useful if you are working with loops in Verilog

```verilog
  integer i;
  always @(posedge clk) begin
    for (i = 0; i < 10; i = i + 1) begin
        // Do something with i
    end
  end
```

See again integers are super useful here for loop indices. They make writing for loops very convenient

Now some extra notes for you here I’ve been asked a lot on this in the past so I have a few extra comments

`integers` are actually 32-bit signed numbers unless specified differently I've never seen the need for other values usually 32-bit signed is enough for my use case and it will most probably be enough for yours too but still you should know this if for example you have a very large memory address space or some very particular high numerical data value then you might need to think and declare appropriately

Also note that while integers are great for looping counters indexing and temporary calculations don't use them to store hardware outputs. For hardware outputs always use `reg` or `wire` types. Integers are just for the simulation and sometimes for some synthesis for some compilers and architectures they might not synthesize correctly but you can use it for most of your regular usage. I remember when I was starting out I made this mistake and the synthesis tools went crazy on me I had registers everywhere and I was wondering where are they coming from. I was very confused and I took me some time to understand where the additional logic came from. It was painful to debug and it was mostly my fault. It is not the tools that are faulty

Another question I get is if integers can be initialized when declared yes you can initialize them it is just as with any other variable in verilog. And it’s very good practice to always initialize your variables if you are doing digital logic design and want to make sure it synthesizes correctly.

For example:

```verilog
integer my_index = 0;
```

This way you know that your counter will start at zero. For counters it’s highly important to initialize them. This also allows to write testbenches with better control as well.

Also keep in mind that integer variables declared inside a always block they are reset every time the `always` block gets called. So you need to initialize them every time if you want that behavior

Also if you ever need to use arithmetic operations make sure that your datatypes are large enough to contain the results of the calculation if it is the case then you might want to use a larger number of bits. If you make mistakes the result will not be correct and it's not always easy to debug it without seeing the error. Sometimes it might work and sometimes not which makes it even harder to debug

Also a good tip avoid mixing blocking and non-blocking statements inside your `always` blocks. It's almost always a very very very bad idea I can't stress this enough especially if you are a beginner it creates so many headache I have been there and I don't want you to be there. You will find it super difficult to debug your code

One more thing I don’t know if it applies but I will say it for my safety when you use integers remember that they are 32 bits signed values so if you do arithmetic you might get a negative value sometimes if it overflows or underflows it's better to always check that.

Speaking of debugging I saw this one time that the issue was with the initialization of an integer variable some other guy had some logic that was expecting some specific value and the code had not initialized this value properly and he was wondering where the issue was I asked him why it is not initialized and he said I don't know and then I helped him and the issue was solved in less than 2 minutes. It was quite funny he said something like it was always an initialization problem. I think this was the most funny thing that ever happened to me in a debugging context. I am still laughing about it.

Now if you want to really master Verilog and hardware design in general, I highly recommend you to read books like "Digital Design and Computer Architecture" by David Harris and Sarah Harris or "Computer Architecture: A Quantitative Approach" by John L. Hennessy and David A. Patterson these are excellent books and they will explain things in great detail.

Also for the specifics of Verilog read "Verilog HDL" by Samir Palnitkar it is like the standard for learning Verilog and understanding all the nuances of the language

There you have it that is basically everything about declaring integers in Verilog. It is not that hard once you understand the basics you can do some awesome hardware stuff and I am sure with time you will master it. And if you have any further questions do not hesitate to ask.
