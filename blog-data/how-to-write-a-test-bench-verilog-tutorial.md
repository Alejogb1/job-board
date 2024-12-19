---
title: "how to write a test bench verilog tutorial?"
date: "2024-12-13"
id: "how-to-write-a-test-bench-verilog-tutorial"
---

Okay so you wanna dive into test benches in Verilog right I get it Been there done that a million times Feels like yesterday I was pulling my hair out with the same problem Back in my uni days we had this as a required course ughh

So lets cut the fluff Test benches are not rocket science theyre really just Verilog modules designed to *test* your other Verilog modules Your design usually the thing you actually care about is the Device Under Test or DUT for short The test bench is the environment you create to see if your DUT behaves as intended Think of it like a science experiment where your DUT is the subject and your test bench is the lab equipment

Now a basic test bench needs a few key things A way to instantiate your DUT a way to provide inputs to the DUT and a way to observe outputs from the DUT Basically you need to stimulate and verify

Let me start with a very basic example just to show the core idea Lets assume you have a simple adder module in `adder.v`

```verilog
module adder(input [3:0] a, input [3:0] b, output [4:0] sum);
  assign sum = a + b;
endmodule
```

Now your basic test bench lets call it `adder_tb.v` would look like this

```verilog
module adder_tb();

  // Parameters
  parameter DELAY = 10; // Simulation time delay unit

  // Inputs
  reg [3:0] a;
  reg [3:0] b;

  // Outputs
  wire [4:0] sum;

  // Instantiate the Device Under Test (DUT)
  adder DUT (
    .a(a),
    .b(b),
    .sum(sum)
  );

  // Stimulus generation
  initial begin
    // Test case 1
    a = 4'b0001;
    b = 4'b0010;
    #DELAY;

    // Test case 2
    a = 4'b0101;
    b = 4'b0110;
    #DELAY;

     // Test case 3
     a = 4'b1111;
     b = 4'b0001;
     #DELAY;

    // Add more test cases here if needed
    $finish; // End the simulation
  end

  // Monitoring (Optional)
  initial begin
    $monitor("Time = %0t a = %b b = %b sum = %b", $time, a, b, sum);
  end

endmodule
```

Okay so what did we do here First we declared inputs as registers `reg` because in a test bench inputs need to be driven You cant directly assign values to wires inside a test bench You also declare the output as a wire like you would in any other module because its still driven by the DUT

Then we instantiated the `adder` module we're testing This part should be familiar if youve done any verilog before The dot notation is just port mapping its important to get this right

The `initial` block is where all the magic happens inside this block you setup your test vectors or test scenarios You have `a` and `b` take on different values and then wait for `DELAY` which is a parameter We use a `parameter` because it allows for flexibility The `#DELAY` syntax advances simulation time which allows you to actually see the results Also note the `#` notation which specifies delays

Finally there is a `$finish` to stop the simulation at the end of the process This also gives a clear endpoint for our test bench We also added a basic `$monitor` to output the values to the simulator console which is good for quick checks

This basic test bench does not verify anything It simply gives the DUT input and lets us observe the output through a console output If you just wanted to check that module worked in basic terms then this could do but if you actually want to check each condition then this is not the way to go

For more complex modules that need more rigorous testing you will need to write more advanced test benches It's all about systematic coverage you have to think like an engineer and anticipate all edge cases and corner conditions This is where we move from simple procedural code to more structured designs in the test bench itself For instance lets move to a Finite State Machine FSM module

Lets say your module is a simple traffic light controller `traffic_light.v`

```verilog
module traffic_light(
  input clk, input reset, output reg [1:0] state
);

  parameter RED = 2'b00;
  parameter YELLOW = 2'b01;
  parameter GREEN = 2'b10;

  reg [1:0] current_state;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      current_state <= RED;
    end else begin
      case (current_state)
        RED: current_state <= YELLOW;
        YELLOW: current_state <= GREEN;
        GREEN: current_state <= RED;
        default: current_state <= RED;
      endcase
    end
  assign state = current_state;
  end
endmodule
```

Now our test bench for this could look something like this `traffic_light_tb.v`

```verilog
module traffic_light_tb();

  parameter CLOCK_PERIOD = 10;
  parameter RESET_TIME = 100;

  reg clk;
  reg reset;
  wire [1:0] state;

  // Instantiation
  traffic_light DUT (
    .clk(clk),
    .reset(reset),
    .state(state)
  );

  // Clock generation
  always begin
    # (CLOCK_PERIOD/2) clk = ~clk;
  end

  initial begin
    clk = 0;
    reset = 1;
    #RESET_TIME;
    reset = 0;

    repeat (10) #CLOCK_PERIOD; // Run for 10 clock cycles

    $finish;

    // You can add more checks or verification logic here
  end

  initial begin
    $monitor("Time = %0t clk=%b reset=%b state=%b",$time, clk, reset,state);
  end

endmodule
```

Now notice a few things we introduced a clock generation using an `always` block This is very common in test benches for synchronous circuits The reset signal is asserted then deasserted after some time The `repeat` statement allows us to run the simulation for a fixed number of clock cycles instead of manually setting up each test case

Also note that the `always` block for clock generation is a non-blocking assignment which we should be wary about as this can create race conditions but in test benches this is okay

Now lets talk about more advanced techniques

Verification using assertion based design with system verilog `assert` is super helpful Here's a simple example lets say we are testing a fifo the code is not complete because of the complexity involved but this will give an idea in a more real scenario of the problems

```verilog
module fifo(
  input clk, input reset, input write_en, input read_en, input [7:0] data_in, output [7:0] data_out, output full, output empty
);
  parameter DEPTH = 8;
  reg [7:0] mem [0:DEPTH-1];
  reg [3:0] write_ptr;
  reg [3:0] read_ptr;
  reg is_full;
  reg is_empty;

  //FIFO logic not included because it is a complicated task to explain everything here

  assign full = is_full;
  assign empty = is_empty;

  assign data_out = mem[read_ptr];

  always @(posedge clk or posedge reset) begin
    if(reset) begin
        write_ptr <= 0;
        read_ptr <= 0;
        is_full <= 0;
        is_empty <= 1;
    end else begin
      // logic here also not completed to keep things simple
    end
  end

endmodule
```

And now our test bench `fifo_tb.v`

```verilog
module fifo_tb();

  parameter CLOCK_PERIOD = 10;
  parameter RESET_TIME = 100;
  parameter DEPTH = 8;

  reg clk;
  reg reset;
  reg write_en;
  reg read_en;
  reg [7:0] data_in;
  wire [7:0] data_out;
  wire full;
  wire empty;

  // Instantiation
  fifo DUT (
    .clk(clk),
    .reset(reset),
    .write_en(write_en),
    .read_en(read_en),
    .data_in(data_in),
    .data_out(data_out),
    .full(full),
    .empty(empty)
  );

  // Clock generation
  always begin
    # (CLOCK_PERIOD/2) clk = ~clk;
  end

  // Test stimulus
  initial begin
    clk = 0;
    reset = 1;
    #RESET_TIME;
    reset = 0;

    // Write some data
    write_en = 1;
    read_en = 0;
    data_in = 8'hAA;
    #CLOCK_PERIOD;

    data_in = 8'hBB;
    #CLOCK_PERIOD;

     data_in = 8'hCC;
    #CLOCK_PERIOD;

    // Verify full flag works
     repeat (DEPTH-3) begin
         data_in = 8'hAA;
        #CLOCK_PERIOD;
      end
    assert (full) else $error("ERROR Full flag is not set");

    // Read Data
    write_en = 0;
    read_en = 1;
    #CLOCK_PERIOD;


    $finish;
  end
    initial begin
      $monitor("Time = %0t clk=%b reset=%b write_en=%b read_en=%b full=%b empty=%b data_in=%h data_out=%h",$time, clk, reset,write_en,read_en,full,empty,data_in,data_out);
  end
endmodule
```

See how we used `assert` to check if the fifo correctly sets the `full` flag If this assertion fails the simulator will output an error message This is so much better than just looking at waveforms or console output because it provides automatic verification of what you are expecting

Also I included more signals in the `$monitor` and this is usually done when testing a more complex module

And remember you should also verify the empty condition works too

Now if you are getting into serious verification with SystemVerilog using constrained random verification is very good It's a whole different ball game which is very complex and it has a very high learning curve so ill skip that for now

But the main idea is that your test bench can be very sophisticated based on the complexity of the module you are testing

Another pro tip for when you have very large designs is to also write scripts in python or tcl to help you compile run and even process results

As for resources for further learning I would suggest the following

*   "SystemVerilog for Verification" by Chris Spear This is a great book if you want to get into more serious verification
*   "Digital Design and Computer Architecture" by David Money Harris and Sarah L Harris This book is more about digital logic and architecture but it has a great overview of the foundations which are very important
*   IEEE 1800-2017 standard document for SystemVerilog This is the bible you should have it open when coding if you want a complete reference for the language
*   Any good textbook on digital circuit design will have a section on testing and verification and its very useful to have them because they will always remind you of the basics

Now a joke for you If a programmer is driving his car and he misses a red light is that a race condition?

Ok i think that's it I tried to put everything I know but I think i got all the points I'm here if you have questions
