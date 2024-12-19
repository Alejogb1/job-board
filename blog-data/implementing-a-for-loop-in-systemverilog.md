---
title: "implementing a for loop in systemverilog?"
date: "2024-12-13"
id: "implementing-a-for-loop-in-systemverilog"
---

Okay so you wanna loop in SystemVerilog right been there done that got the t-shirt and probably spilled coffee on it too more times than I'd like to admit Let's break this down it's not rocket science but it's got its quirks

See SystemVerilog isn't your average C++ or Python it's hardware description language at its core so looping isn't like banging out some quick iterations it's about describing hardware that will do the looping for you that's a big difference and a big mindset shift

My history with loops in SystemVerilog goes back to my early days doing some FPGA design I remember this one project it was a digital signal processor well part of one anyway I needed a pipeline stage that required multiple arithmetic operations on data array. I thought "Great loops are gonna be my best friend". Reality hit me like a brick of Verilog simulation errors. Turns out hardware and software loops are way different. I was creating combinatorial logic nightmares instead of the sequential ones I wanted. This was not good. So let’s not fall into the same trap yeah?

The basic structure you are looking for is the classic for loop. I'll give you the basic syntax

```systemverilog
for (initialization; condition; step)
begin
  // code to execute
end
```

Easy right? Sort of But the details matter a lot in this world

*   **Initialization:** This is where you declare and often set your loop counter variable It is good practice to declare the iterator variable within the for statement but you don't have to. It's scoped to the loop.
*   **Condition:** This is where the loop decides whether to keep going. If the condition is true it continues if false the loop is done. Simple enough.
*   **Step:** This determines what happens to the loop counter after every iteration usually an increment or decrement.

Let's get concrete This is an example of a basic loop that increments a value across a vector called *output_data*. It iterates through 16 values using `i` as a simple counter

```systemverilog
module for_loop_example (
  output logic [7:0] output_data [15:0] //16 x 8 bit array
);

  integer i;
  always @(*) begin
    for (i = 0; i < 16; i++) begin
        output_data[i] = i;
    end
  end
endmodule
```

Okay few things. First the `integer i` bit is the standard way to define a loop counter variable here We use `@(*)` because this is combinational logic. This loop is not happening in time it will create a circuit and the variable "i" will change continuously with respect to whatever inputs the circuit has (nothing in this case but the value will be different if the code was inside a bigger module and inputs existed).

Second, the `output_data[i] = i;` part does what it says its gonna assign a value to the output array it will assign zero to `output_data[0]` one to `output_data[1]` and so on.

But hold on we’re not done with the caveats

One big mistake I see people make is trying to do loops inside clocked always blocks for example without understanding the implication of synthesizable code.

If you are in a clocked `always @(posedge clk)` block then the loop needs to be *unrolled* when it's synthesized into hardware. The way we get the correct and synthesizable loop behavior with clocked sequential circuits is by creating a state machine. The loop variable needs to become a register and that needs to be controlled by sequential circuits this way we have correct values during the operation of the loop.

Let's look at another example with a `posedge clk` and an input called *input\_data* and a variable called *sum*. The goal is to sum up the vector *input\_data* and store it inside *sum*.

```systemverilog
module clocked_for_loop (
  input  logic clk,
  input  logic [7:0] input_data [15:0], // 16 x 8-bit array
  output logic [15:0] sum
);
  integer i;
  reg [15:0] sum_reg;
  reg [3:0] state;

  parameter IDLE = 4'b0000;
  parameter LOOP = 4'b0001;
  parameter DONE = 4'b0010;

  always @(posedge clk) begin
    case (state)
        IDLE: begin
            sum_reg <= 16'h0000;
            i <= 0;
            state <= LOOP;
        end
        LOOP: begin
             if (i < 16) begin
                 sum_reg <= sum_reg + input_data[i];
                 i <= i + 1;
             end
             else begin
                state <= DONE;
             end
        end
        DONE: begin
          // do nothing
         end
        default: begin
        // something wrong 
        state <= IDLE;
        end
    endcase
  end

    always @(*) begin
    sum = sum_reg;
    end
endmodule
```

What did we do here? We created a state machine with `IDLE`, `LOOP`, and `DONE` states. `IDLE` initializes the loop and sets the state to `LOOP`. `LOOP` does the actual addition and when `i` reaches 16 it will proceed to the `DONE` state. This state doesn’t do anything to it's just so we can keep the value inside the `sum` output.

Another tricky thing to remember SystemVerilog loops are usually executed as part of the synthesis step so don't expect a dynamically sized loop based on run-time variables in the general case. The size and shape are generally determined before the synthesis happens. The synthesizable code means that these loops need to be constant. This means that the number of iterations needs to be known in advance and at compile time so the synthesizer can unroll that logic.

Another useful loop construct that can be useful in SystemVerilog is the *foreach* loop, which is great for iterating through array elements when you are not using hardware but are using SystemVerilog to verify your design. This is not part of the hardware that will be synthesized. It’s part of the verification environment. Let's see the example of this kind of loop

```systemverilog
module verification_loop();

  logic [7:0] data [5:0] = '{8'h01, 8'h02, 8'h03, 8'h04, 8'h05, 8'h06};
  integer i;

  initial begin
    foreach (data[i]) begin
      $display("Data[%0d] = %0h", i, data[i]);
    end
  end
endmodule

```

This will print the values of the array to the console one by one. `foreach` does the iteration and you get the values of the indexes inside `i`. This is a testbench level tool and is a great example of the *verification features* that systemverilog gives you.

Now I know what you are thinking "How can I learn more about this black magic". You won't find it on page 33 of some manual. Here are some of my recommendations based on my experience:

*   **"SystemVerilog for Design" by Stuart Sutherland:** This is a classic it’s the bible for SystemVerilog it goes deep into all the details you need when thinking about implementation. I've had this book since forever.
*   **"Verification Methodology Manual" (VMM):** This one is really good for understanding the verification concepts specifically but it does go into SystemVerilog concepts that are useful for implementation as well. It is a methodology bible for verification.
*   **"Digital Design and Computer Architecture" by David Money Harris and Sarah L Harris:** This one is not specifically about SystemVerilog but this is the book that helped me understand the hardware side of things. It's good to know what's happening inside hardware after you synthesize your code.

Oh and one last thing a bit of personal advice if you get stuck with a for loop issue in SystemVerilog and spend too much time you have to start looking at the bigger design picture. Sometimes you have to rethink things maybe you are using a tool that is not appropriate for the task and then you will find the solution when least expect it. I once was stuck with a for loop for an entire afternoon it turns out I could avoid the loop all together by using better memory structure... It was one of those “Doh” moments.

And that's it for for loops in SystemVerilog. I really hope that you don’t have to go through the struggles I had when starting with this. Now go code something awesome!
