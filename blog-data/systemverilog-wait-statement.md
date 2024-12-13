---
title: "systemverilog wait statement?"
date: "2024-12-13"
id: "systemverilog-wait-statement"
---

Okay so you’re asking about `wait` in SystemVerilog huh Been there done that got the t-shirt literally I have a drawer full of vendor swag with subtle wait statement horror stories embroidered on them I kid you not

Let’s break it down no fluff just the technical bits `wait` is like the basic building block of synchronization in SystemVerilog a way to make sure your testbench or your verification environment doesn't go off the rails and start doing things out of order It's about telling a process to chill out for a bit until a specific condition becomes true

Now the basic syntax is dead simple `wait (expression);` You’ve got your keyword then you wrap the thing you’re waiting for in parenthesis Pretty straightforward but it's like a simple tool that can cut your finger off if you don't use it correctly So lets discuss it more in detail lets start with the expressions that you can use there

That expression can be anything that resolves to a boolean it's true or false I mean a simple variable a signal going high or low or a more complex logical operation anything really that the simulator can evaluate at a given time

Here's the thing though this isn't just a static pause It's an active wait The simulator's watching that expression like a hawk if that expression is false it freezes that process's execution until the expression becomes true and then it resumes from where it left off

Been there many times believe me first time I used wait I thought everything was fine but then during my simulation some processes where going in the wrong order because I didn't fully understand how it works And lets be real no one fully understands it at the start we all learn it through trial and lots of debugging hours.

Lets go through some common cases I've seen in my career

First a basic signal transition example This is like your bread and butter stuff and its the most common use I have used the `wait` statement for:

```systemverilog
module wait_example_1;
  logic clk;
  logic reset;
  logic data_ready;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    reset = 1;
    #10 reset = 0;
  end

  initial begin
    #20 data_ready = 1;
    #20 data_ready = 0;
  end

  initial begin
    wait(reset==0); //wait until reset goes low
    $display("Reset Done");
    wait (data_ready);  // wait until data_ready is high
    $display("Data ready signal seen");
   #10;
   wait (!data_ready); //wait data_ready signal to go low
   $display("Data ready signal is low now");
  end
endmodule
```

Here the initial process is paused at the `wait(reset==0);` line until the reset signal goes low then it prints the reset done message then the code moves on to the next `wait` statement then it waits until the data ready signal goes high and when that happens it executes the next line and so on Simple as that

But the beauty in `wait` comes when you need to synchronize with other processes think about something like a handshaking mechanism between two modules The `wait` here ensures that data is ready before another module starts its processing

Example 2: Synchronizing two modules

```systemverilog
module handshake_example;
    logic req;
    logic ack;
    logic data;

    initial begin
    $display("Start of process one");
    req = 1;
    #10 data=1;
    $display("Process 1 finished and requesting data ");
    wait(ack==1);
    $display("Process 1 got the acknowledgment proceeding with computation");
    #10 $finish;

    end

    initial begin
      $display("Start of process two");
      wait (req==1);
      $display("Process 2 received a request and processing");
      #10;
      ack = 1;
      $display("Process 2 Acknowledgment send back");
      #20 $finish;
    end
endmodule
```

In this case the first initial block is waiting for the ack signal while the second initial block is sending the ack signal and they do this based on a request signal that the first initial block is sending making them synchronized to perform specific task in a determined order

Now the real fun starts when you get into complex conditional waits You can build some really specific conditions for `wait` its not just waiting for one single signal to change.

Example 3 Complex conditional waits

```systemverilog
module conditional_wait_example;
    logic signal_a;
    logic signal_b;
    logic signal_c;

    initial begin
      #10 signal_a = 1;
      #10 signal_b = 1;
      #20 signal_c = 1;
      #20 signal_a= 0;
      #20 signal_b = 0;
      #100 $finish;
    end

    initial begin
      wait(signal_a && signal_b && signal_c);
      $display("All signals are high ");

      wait(!signal_a || !signal_b);
      $display("Signal A or B is low");

    end

endmodule
```

Here, the first wait waits until `signal_a`, `signal_b`, and `signal_c` are all high. The second wait is executed when either `signal_a` or `signal_b` goes low. It allows you to express more intricate synchronization logic directly in your code. This is helpful when you need very specific conditions for your testbench to behave as expected.

The thing that many people forget though is that `wait` is a blocking statement It’s like when you try to use a public restroom and someone is taking a long time in there it stops the current process's execution That can be a bit of a problem if you have a very long condition in your wait or if you’re waiting for something that never happens.

The simulator stops on that `wait` line waiting for the expression to go high It's crucial to make sure your waiting condition will eventually become true or you are going to have a simulation stall it can even stall the entire thing believe me I have seen that happening many times and it always makes me cry a bit inside because it is like I am in a deadlock situation (not literally of course hehe)

One common mistake I've seen is people using wait in place where they should be using event triggers or non blocking assignments in some cases event triggers provide more fine control over the process execution In another scenario using blocking assignments instead of non blocking assignments can create race conditions so it's really important to know the limitations of the language

Now a pro tip from my experience I find it helpful to put a timeout in a `wait` for that kind of situation That way your simulation does not get stuck in a infinite wait loop if something doesn't happen as planned. You can do this with a combination of delay and `wait`

So instead of doing this `wait(some_condition);` you can do something like this
```systemverilog
#timeout_value;
if (!some_condition) begin
  $display("Timeout occurred condition was not met");
end else begin
  wait(some_condition);
  $display("Condition met");
end
```

This way if the timeout is reached the process will exit and the simulation can proceed. This is useful for catching errors in your testbench.

If you want a more in depth understanding about SystemVerilog there are great books that I can recommend I am always learning more about it even after many years working with the language.  There's "SystemVerilog for Verification" by Chris Spear and Greg Tumbush It’s like the Bible for verification folks its a great book if you want to dive deep in the subject Also if you want to learn more about the language itself and also about assertions I would recommend you to check "Writing Testbenches using SystemVerilog" by Janick Bergeron it is also a very nice book.

So to wrap it up `wait` is that workhorse of SystemVerilog synchronization but like any workhorse you need to know its limits and how to use it wisely its easy to learn but you have to master it to use it correctly so don't use it without fully understand its implication in your simulation process. Take your time learn from your mistakes and your simulations and you will be a pro using the `wait` statements before you even realize it. Just keep practicing and you will be a `wait` statement expert in no time.
