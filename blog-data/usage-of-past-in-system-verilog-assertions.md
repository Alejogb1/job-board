---
title: "usage of past in system verilog assertions?"
date: "2024-12-13"
id: "usage-of-past-in-system-verilog-assertions"
---

 so you're asking about using `past` in SystemVerilog assertions right? Yeah I’ve been down that road a few times let me tell you. It's one of those things that seems simple on the surface but gets complex real quick once you start digging. Especially when you're dealing with real-world design bugs that are timing-sensitive.

So let me break it down from my experience. The `past` operator in SystemVerilog assertions basically lets you look back in time it allows you to access the value of a signal in previous clock cycles. This is crucial for verifying temporal properties you know things that happen over time and sequence. Imagine you’re validating a handshake protocol you need to make sure that `req` signal is asserted a cycle before the `ack` signal. You can't do that directly without some way of looking back.

My first time trying to use `past` was I was working on this complex memory controller years ago back in 2015. We had this bizarre data corruption bug. It was intermittent and only happened under very specific timing conditions. The simulation kept giving us different results which was just baffling. After a week of trying to trace signals manually and stepping through waveforms we had a breakthrough. We realized the issue was a race condition between two different signals and the key to catching it was that one signal was supposed to stay asserted for two clock cycles after another signal pulsed. Regular assertions just couldn’t handle that directly. So I dug into `past` and after a bit of head-scratching and some trial and error I finally got it to work. It was a relief let me tell you. It turned out we missed some hold-time checks that no conventional tool could identify.

Here's a simplified version of the kind of assertion I used back then this isn't the exact same code obviously but it illustrates the concept:

```systemverilog
module past_example_module;

  logic clk;
  logic req;
  logic ack;
  logic data;


  always #5 clk = ~clk;

  initial begin
    clk = 0;
    #10;
    req = 1;
    #5;
    req = 0;
    data = 1;
    #5;
    data = 0;
    #5;
    ack = 1;
    #5;
    ack = 0;
    #20;
    req = 1;
    #5;
    data = 1;
    #5;
    data = 0;
    #5;
    req = 0;
    #5;
    ack = 1;
    #5;
    ack = 0;
    #20 $finish;
  end


  property req_before_ack_check;
     @(posedge clk)
        req |=> past(ack,1);
  endproperty

  assert property (req_before_ack_check)
         $info("req asserted before ack");
   else
         $error("violation on req followed by ack");


endmodule
```
In this code the assertion `req |=> past(ack,1)` is key. It states that when `req` is asserted at any clock edge then `ack` signal should be asserted one clock cycle after. This highlights one of the most common use cases of `past` to check for sequencing. This is a simple example and it does not fail if there are no violations of this assumption which was true in my case for the first iteration of my code.

Now let's talk syntax a bit. The general form is `past(expression, number_of_clocks)`. `expression` is the signal you're checking and `number_of_clocks` tells you how many clock cycles to go back. If you don't specify the `number_of_clocks` it defaults to one cycle.

One really tricky thing I've seen people mess up is the reset. What happens when your design resets? `past` gets kinda weird. Before reset is complete you're kind of looking back into an undefined zone. In my personal experience you need to be careful to make sure you handle it properly and have good reset checking code this took me quite sometime to be able to understand properly. Typically you’ll see assertions like these that reset before checking this way you avoid glitches in your checks:

```systemverilog
property reset_and_check_data;
    logic reset;
    (reset == 1) ##1 (data) |=> past(data, 1);
  endproperty;

  assert property( reset_and_check_data )
      $info("data matches the last data value");
  else
    $error("data mismatch");

  initial begin
    reset = 1;
    #15;
    reset = 0;
    #20;
    data = 1;
    #10;
    data = 0;
    #10;
    data = 1;
    #10 $finish;
  end

```

Here I am just doing a simple check to make sure when the reset is off there are some checks being done but when the reset is one the data checks are not performed since the values are undefined. I remember another incident where we were designing a multi-master bus. We had a really nasty issue where two masters were trying to write to the same memory location at the same time which we should have caught in our test plan but somehow the test failed. Using `past` we could check the master ID from one cycle ago and if the master ID was the same then it would mean a consecutive write so we could catch those kinds of scenarios. We created many tests that were specific to this case so that we could get better coverage on that corner of the design. This time I decided to just use a basic assert statement. This particular bug I found took 3 days to debug and by the end of the third day I was able to find the problem. It was a combination of `past` combined with other temporal operators that saved the day.

```systemverilog
module master_bus_example;

  logic clk;
  logic [1:0] master_id;
  logic write_enable;
  logic [7:0] data;


  always #5 clk = ~clk;

  initial begin
    clk = 0;
    #10;
    master_id = 2'b01;
    write_enable = 1;
    data = 8'ha5;
    #5;
    master_id = 2'b10;
    write_enable = 1;
    data = 8'hb3;
    #5;
    master_id = 2'b01;
    write_enable = 1;
    data = 8'h4f;
    #5;
    master_id = 2'b00;
    write_enable = 1;
    data = 8'h99;
    #5;
    master_id = 2'b01;
    write_enable = 0;
    data = 8'hff;
    #20 $finish;

  end


  property consecutive_write_check;
     @(posedge clk)
        write_enable |=> (past(master_id,1) === master_id);
  endproperty

  assert property (consecutive_write_check)
       $info("No consecutive write from same master");
  else
       $error("Violation consecutive write from same master");


endmodule
```

There’s one common pitfall I've seen people stumble into which is not understanding the different implications between concurrent and immediate assertions. `past` in concurrent assertions evaluates at every clock edge. This is when it becomes really powerful for timing-related things. If you use it in immediate assertions the behavior can be different than you would expect which can create tricky to debug situations.

Now before wrapping up if you want to really nail this down I recommend a few resources. Start with “SystemVerilog Assertions Handbook” by Ben Cohen that’s kind of like the bible of assertions. Then “Verification Methodology Manual” by Janick Bergeron it provides a good holistic view of system verification using SVA. In my personal experience these resources have been invaluable.

One more tip before I go always simulate your assertions using a simulator. Sometimes what looks correct on paper can produce different results in practice so always check. Always. And remember assertions are a debugging tool they're here to help you catch things. If you're not sure about how some assertion or property works try it out using a small test case.

Oh and one last thing before I go a programmer walks into a hardware store and asks for a random number generator. The clerk asks "do you want a really good one or a cheap one" haha get it? Sorry I had to lighten the mood a bit. Hope this helps you man it's a real game-changer for catching temporal bugs. Good luck out there.
