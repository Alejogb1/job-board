---
title: "generate clock in verilog code?"
date: "2024-12-13"
id: "generate-clock-in-verilog-code"
---

Okay so you want a clock generator in Verilog right I've been there man I mean everyone's been there if you've ever touched an FPGA or ASIC for more than five minutes It's like the hello world of hardware description languages

So first off let's get this straight a clock isn't just some magical thing that happens it's a signal a very specific oscillating signal that we use to synchronize operations in digital circuits. Think of it as the heartbeat of your system Without it everything's just a bunch of logic gates doing their thing randomly and that's not useful at all. You need that rhythm to make things happen in the correct sequence.

Now you could be asking about a few things here we'll go with the simplest which is just a basic clock generator usually for testing or simulations. Something you’d see in a testbench not an actual design. You don't normally create a raw clock generator in an actual hardware implementation. You usually get those from dedicated clock modules often with PLLs inside. But for simulations yeah I've had to do this a bunch.

So here’s your basic clock generator code something that alternates between 0 and 1 every so often:

```verilog
module clock_generator #(parameter CLK_PERIOD = 10) (output reg clk);

  initial begin
    clk = 0;
  end

  always #(CLK_PERIOD/2)
    clk = ~clk;

endmodule
```

This snippet defines a module called `clock_generator`. It has one parameter `CLK_PERIOD` which defaults to 10. This parameter will determine how fast your clock oscillates this is obviously in simulation time. It has a single output called `clk` which is a register. Initially the `clk` register is set to 0. And the `always` block does the magic toggling it with a delay. The `#` operator is how you add delays in Verilog and the divide by two is needed because the delay is half the total period.

This code is fine for a simple simulation but you need to remember that a real clock signal will have a certain rise time and fall time and it might not be a perfect square wave. Also in this case, if your simulation time steps are smaller than the half-period delay, you could experience issues. But for basic testing this is fine. I’ve used this so many times it’s not even funny. Actually it is kind of funny, considering how many hours I spent figuring this out. It's like the first time you try to make instant noodles and you somehow burn them. We've all been there.

Now what if you need a more specific frequency control lets say you want to derive a lower frequency from a high frequency system clock this is where dividers come in handy. So you can also make the clock generator more complex here is a little example of a clock divider:

```verilog
module clock_divider #(parameter DIVISOR = 2) (input clk_in, output reg clk_out);

  reg [31:0] counter;

  initial begin
    counter = 0;
  end

  always @(posedge clk_in) begin
    if (counter == DIVISOR - 1) begin
      counter <= 0;
      clk_out <= ~clk_out;
    end
    else begin
      counter <= counter + 1;
    end
  end

endmodule
```

This `clock_divider` module takes an input clock `clk_in` and generates a new clock `clk_out` at a lower frequency specified by the `DIVISOR` parameter.  The `counter` register counts the positive clock edges of the incoming `clk_in` clock until it reaches `DIVISOR - 1` when it toggles the output `clk_out` and resets the counter back to zero.

Let's say your incoming clock is 100 MHz and you want a 50 MHz clock out you’d use a `DIVISOR` of 2. For 25 MHz use 4 and so on. It’s just simple division. Just be aware that this type of divider can’t create arbitrary frequencies you’re limited by the divisor you use and you can only go lower never higher this is a very important aspect of clock dividers also this approach will introduce jitter.

A real-world clock would also have variations in the rise and fall times these are typically modeled using dedicated models but you won’t really do that in a simple testbench clock. Usually when you get to the actual verification process you are going to be using a dedicated clock generator module with configurable parameters.

Finally for more complex simulation scenarios or if you want to generate multiple clock signals you might find it useful to model a clock management unit (CMU) so here is a small example of that:

```verilog
module clock_management_unit #(parameter FREQ1 = 10, parameter FREQ2 = 20) (output reg clk1, output reg clk2);

  reg [31:0] counter1;
  reg [31:0] counter2;

  initial begin
    counter1 = 0;
    counter2 = 0;
    clk1 = 0;
    clk2 = 0;
  end


  always #1 begin
    counter1 <= counter1 + 1;
    if (counter1 >= FREQ1) begin
      clk1 <= ~clk1;
      counter1 <= 0;
    end
    counter2 <= counter2 + 1;
    if (counter2 >= FREQ2) begin
        clk2 <= ~clk2;
        counter2 <= 0;
    end
  end

endmodule
```
This `clock_management_unit` module creates two clocks `clk1` and `clk2` with different frequencies specified by the parameters `FREQ1` and `FREQ2`. This approach is a bit different than the divider but in the end achieves a similar result. Note that these parameters are used as a threshold counter when it is reached the counters are reset and the clock output is toggled. This way we can emulate two different clocks with different periods.

I've used a bunch of ways to generate clocks but honestly these three examples will get you through the majority of the situations. Again real life you use those special clock modules that take care of PLLs and skew but for simulation purposes this is good enough.

If you want to dive deeper I would highly recommend reading up on digital design principles. Textbooks like "Digital Design and Computer Architecture" by Harris and Harris or "Computer Organization and Design" by Patterson and Hennessy are really good resources for understanding how clocks actually work in digital circuits and how you can deal with issues like clock skew and jitter. They cover topics from the basics of logic gates all the way to complex processor designs. Also, the documentation provided with your EDA tools has in-depth explanations about clock modeling. It is worth spending some time there.

These codes snippets can help you to get you started but like every coding experience it depends on the situation and you will need to modify them to your specific needs. This code isn’t production-ready this is for simulations and simple designs only. Remember simulation is not the same as actual hardware you always need to consider this point during design. Good luck.
