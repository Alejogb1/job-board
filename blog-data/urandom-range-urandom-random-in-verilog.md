---
title: "urandom range urandom random in verilog?"
date: "2024-12-13"
id: "urandom-range-urandom-random-in-verilog"
---

Okay so you're asking about generating random numbers in Verilog using something like a urandom range or urandom random right Been there done that more times than I care to admit Let's dive in

First off let me tell you my pain points with Verilog and random numbers I was working on this massive chip a few years back for a high performance data processing system and we needed really good pseudo random number generators or PRNGs for testing the memory controllers Like not just something that vaguely looks random but actually good for stress testing with a uniform distribution and minimal repetition What I initially tried was just relying on the plain `random` system function provided by Verilog simulators Oh boy was that a mistake

The initial tests with that naive approach looked okay on a small scale but as I scaled up the test sizes the issues started to become glaringly obvious My memory controller testing was showing all sorts of non random patterns and strange clustering in terms of which addresses it accessed basically it was far from uniformly random My debug session stretched longer than an episode of star trek I was so tired I was probably close to a borg but with even worse assimilation skills

It turns out the built in `random` function in older simulators sometimes doesn't generate a really high quality random sequence And this is an understatement It's often tied to some fixed seed or relies on pretty simple algorithms so yeah I learned the hard way that the term random in Verilog is sometimes like a random dog at the park doing its own thing

So yeah I shifted to using SystemVerilog where you have access to `urandom` and `urandom_range` These are way better for generating good pseudo random numbers in a simulation environment They are based on better algorithms and give you more control over the generated sequence and distribution

Let's get to the specifics `urandom` itself returns a 32 bit unsigned random number This is a good general purpose PRNG you can use if you dont need to constraint your results or don't care about range limits

Here's an example snippet:

```verilog
module random_example;
  reg [31:0] rand_num;

  initial begin
    $display("Random Number Tests");
    repeat(5) begin
      rand_num = $urandom();
      $display("Random Number: %h", rand_num);
      #1;
    end
  end
endmodule
```

That code will print out five different 32 bit random numbers in hex format If you run it repeatedly each time you will get a different sequence of number I know it sounds simple but the impact of moving away from the plain random function cannot be understated

Now if you need to generate a number within a specific range that is where `urandom_range` shines

Here's an example demonstrating how to use `urandom_range` where a random number will be generated between a lower and upper value in this case between 10 to 20 inclusive

```verilog
module range_example;
  reg [31:0] range_num;
  integer lower_limit = 10;
  integer upper_limit = 20;

  initial begin
    $display("Random Number within range");
    repeat(5) begin
       range_num = $urandom_range(upper_limit, lower_limit);
       $display("Random Number in range %0d to %0d : %d", lower_limit, upper_limit, range_num);
      #1;
    end
  end
endmodule
```

Notice that in the `urandom_range` the upper limit is given first followed by the lower limit it is something you need to keep in mind because it is the opposite of the usual intuition that we are accustomed to

One other important point is when you use urandom and urandom range the number generated is pseudo random so if you start the simulator with the same seed or without a seed it will result in the same random pattern That might be a desirable effect for debugging and reproducing a certain test scenario but in other cases you need to be able to randomize the seeds this will be achieved by using something like `$time` as a seed

Here is an example on how to randomize seeds:

```verilog
module seed_example;
    reg [31:0] rand_num;
    integer seed;

    initial begin
        $display("Random Numbers with Randomized Seed");

       seed = $time;
       $srandom(seed);

       repeat(5) begin
            rand_num = $urandom();
            $display("Random Number: %h", rand_num);
             #1;
       end
    end
endmodule
```

In this code, the current simulation time `$time` is used to seed the pseudo random generator this guarantees that every time the simulation starts a different sequence of random numbers will be generated unless you start the simulator at precisely the same time value which is very unlikely in practice

You might be wondering now which kind of PRNG algorithm they use and what are the best practice for using them well that is a deeper subject to delve into I strongly recommend to have a read on these materials which are excellent references that go deep into the theory and the implementation: Donald Knuth "The Art of Computer Programming Volume 2: Seminumerical Algorithms" this is a classical reference book that has a section on pseudo random number generators or if you want something slightly more specialized then I highly recommend “Handbook of Applied Cryptography” by Alfred J. Menezes Paul C. van Oorschot and Scott A. Vanstone. It's not just about cryptography it has a great section on random number generation. These are a great resource if you want to get in the details of the theory and different algorithms as well as the limitations of those different kind of random algorithms

Also a personal tip always double check how your Verilog simulator handles random seeds and the behavior of the `$urandom` function across different versions It is a good idea to verify that using statistical tests if you are using a critical application so you dont end up in a similar situation that I had with my non random memory tests. Also a funny thing that happened to a colleague once was that he was testing a simulation that did not behave as it should he spend a lot of time looking for a bug in the verilog but it was ultimately that a hardware glitch was causing the faulty behavior it felt like the code was so random it was mocking us. So yeah pay attention to all layers not only to the Verilog.

In summary for good pseudo random numbers in SystemVerilog always prefer `urandom` and `urandom_range` over the old `random` use a random or changing seed and read those books I suggested. Hope that helps and happy coding
