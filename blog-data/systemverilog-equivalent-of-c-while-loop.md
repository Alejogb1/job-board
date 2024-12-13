---
title: "systemverilog equivalent of c while loop?"
date: "2024-12-13"
id: "systemverilog-equivalent-of-c-while-loop"
---

Okay so you want to know the SystemVerilog equivalent of a C `while` loop right I get that it's a pretty basic concept but sometimes the syntax differences between languages can throw you off especially when you're used to C and diving into hardware description languages like SystemVerilog

I remember the first time I had to translate a C algorithm into SystemVerilog for a digital signal processing project back in my university days it was a nightmare I kept getting synthesis errors that made no sense I spent a whole weekend just debugging a seemingly simple loop because I was mixing up the syntax between C and SystemVerilog It was one of those moments where I questioned my life choices but yeah we all have been there

The short answer is yes SystemVerilog has its own version of `while` loop but it comes with some caveats due to the nature of hardware design unlike C where code executes sequentially in a single thread SystemVerilog often targets parallel hardware so the while loop needs to be understood in a slightly different context

Let's break it down think of a while loop in C like a continuous checking process If the condition is true the loop body executes and then the condition is checked again until it is false

```c
// C example
int i = 0;
while(i < 10) {
    printf("Count %d\n", i);
    i++;
}
```

In C this executes sequentially incrementing 'i' and printing until 'i' is no longer less than 10 It's predictable because it's running step by step

Now in SystemVerilog we often use while loops within a process or a always block and how you use it is very important especially if it runs inside logic which does the heavy lifting or if you have to use timing control using delays

The general SystemVerilog syntax for a while loop looks identical to C's

```systemverilog
// SystemVerilog Example
module while_loop_example;

  reg [3:0] i;

  initial begin
    i = 0;
    while (i < 10) begin
      $display("Count %0d", i);
      i = i + 1;
    end
  end

endmodule
```

Notice how similar it is to the C example above This code will simulate printing numbers from 0 to 9 but here is the kicker a crucial difference this SystemVerilog `while` is not intended to describe a real-time continuous hardware circuit It's more appropriate for testbenches or initial blocks for simulation setup or verification that is the key difference we need to take into account

If you want to synthesize hardware from a loop you need to be aware of loop unrolling which is not something you need to concern in C programming or if you have you do it manually In simple terms a synthesis tool might choose to duplicate hardware logic to perform the steps inside the loop so you want to limit the complexity of the loop for hardware generation and try to not use variables which grow in complexity with every loop

Now let's try an example with a little bit of a challenge if you want to try it on your own which we can include a delay and see how it goes

```systemverilog
module while_loop_with_delay;

    reg [3:0] counter;
    reg clk;

    always #5 clk = ~clk; // 5 time unit clock period

    initial begin
        clk = 0;
        counter = 0;
        #10; // Delay to let clock start
        while(counter < 10) begin
            @(posedge clk); // Wait for positive clock edge
            $display("Counter value: %0d at time %0t", counter, $time);
            counter = counter + 1;
        end
        $finish;
    end

endmodule

```

This example showcases a more practical use of a `while` loop with a clock event control this is similar to how you might use a clock signal to control when you take the action based on external timing source like a real system would The `@(posedge clk)` is not something you see in C but it's essential in SystemVerilog for timing controlled simulation and this time with positive edge which is common but you can use the negative edge or both edges if you want to

Here's the thing I've had my share of frustrating moments with loops in hardware design I remember back in a project where I was designing a simple finite state machine FSM I decided to use a while loop to set the FSM state transitions inside an always block which seemed logical at the time And it didn't work I had to use case statements and other techniques to get it done properly which was a bit embarrassing for me but hey experience is a teacher who charges tuition you know

The loop was trying to do too much all at once and the synthesizer had no idea what to do with it it was a long debugging session I learned a lot about synthesis limitations from that experience especially about combinational loops which are an absolute no no for synthesis

One point that tripped me up several times was the difference between simulation and synthesis in simple words not everything that works in simulation will work in the synthesis and the other way around so you have to think from the hardware perspective too when you design a circuit

I guess what I’m saying is that while a `while` loop is simple in C think of it in SystemVerilog as a versatile tool for simulation and verification but you also have to understand its implications when you are using it for synthesizable logic There are other loops like 'for' and 'foreach' which may be better options for some situations

Just remember that `while` loops can be powerful for creating test benches but when generating hardware you should always keep the synthesis implications in mind

If you want a good resource I’d suggest checking out "SystemVerilog for Verification" by Chris Spear it covers these kinds of nuances well and helped me a ton Also there are some online resources but be mindful not all of them are accurate I also found "Verification Methodology Manual" aka the VMM helpful but it is quite long book so its your choice which to pick but the first one is a good starter I would say

Here's one more thing about my personal experiences with Verilog like languages. I always say that it is better to start simple and incrementally add complexity because otherwise you will get lost on debugging or even understanding what's going on. I mean, come on who hasn't written a 500 line Verilog file only to have the simulator scream back at you with one obscure error message. It's like debugging code when you don't have a debugger which I had to do in some ancient devices when working in old tech companies it was a time when we all had to understand the hardware because of our life depend on it. And believe me those times were hard. But that's a story for another time

Oh just before I forget I have a small joke for you. Why do programmers prefer dark mode? Because light attracts bugs haha yeah I know it is a stupid joke but I am a bit sleep deprived here I will get a coffee and get on with my tasks

Anyway I hope this clears the air for you and if you have more questions just fire them out
