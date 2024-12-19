---
title: "using enum in verilog?"
date: "2024-12-13"
id: "using-enum-in-verilog"
---

Okay so you wanna talk about enums in Verilog right? Alright been there done that got the t-shirt and probably a few scars too. Let me tell you I’ve wrestled with this beast enough to fill a small notebook. Let's get into it.

So yeah using `enum` in Verilog is something I see people struggle with all the time and honestly I used to struggle too way back when. It's not like some languages where you can just declare an enum and everything works magically. Verilog’s a bit more… deliberate you might say. You gotta be explicit or you’re gonna have a bad time trust me on this one.

First thing to understand is that Verilog is primarily a hardware description language not a high-level software language. It's designed to model hardware structures not abstract data types. So enums in Verilog are really just a way to assign symbolic names to integer values. It's syntactic sugar to make your code more readable maintainable and less prone to errors. You're not creating some fancy new data type. You’re just mapping names to numbers. This tripped me up so many times back in the day. I remember pulling all-nighters on a project and debugging random values propagating through the design because I mixed up numbers and thought of an enum like a strongly typed variable from a different language. Man that was painful.

Now the `enum` keyword itself is a relatively recent addition it wasn’t there in the older Verilog standards. So if you’re working on older code or with tools that don't support it you will not find it and you will probably need to manually define those state values for example using `parameter`. I spent like a week debugging one particular chip because the compiler they were using didn't support `enum` and we did a manual replacement which had a small typo. It was maddening.

Here’s a basic example of how you might use an `enum` in Verilog:

```verilog
module enum_example;

  typedef enum {
    STATE_IDLE,
    STATE_READ,
    STATE_WRITE,
    STATE_DONE
  } state_t;

  reg [1:0] current_state;  // Must be large enough to hold all enum values
  
  always @(*) begin
    case (current_state)
      STATE_IDLE : $display("State is IDLE");
      STATE_READ : $display("State is READ");
      STATE_WRITE : $display("State is WRITE");
      STATE_DONE : $display("State is DONE");
      default : $display("Invalid State");
    endcase
  end
  
  initial begin
    current_state = STATE_IDLE;
    #10 current_state = STATE_READ;
    #10 current_state = STATE_WRITE;
    #10 current_state = STATE_DONE;
    #10 current_state = 4; // This would be considered invalid

  end

endmodule
```

Notice the `typedef enum` declaration This creates a new type called `state_t` which can hold the states. Each state gets an implicit integer value starting from 0. So `STATE_IDLE` is 0 `STATE_READ` is 1 and so on. You can also explicitly assign values to each item if you want.

And importantly you still have to define a storage variable with a certain number of bits that can hold the maximum number of states. This is super important.

Here’s another example where I explicitly assign values. It’s useful if you need to match the bit patterns with a specific protocol or external device:

```verilog
module enum_explicit_value;

  typedef enum {
    CMD_RESET  = 2'b00,
    CMD_READ  = 2'b01,
    CMD_WRITE = 2'b10,
    CMD_STATUS = 2'b11
  } command_t;

  reg [1:0] current_command;

  always @(*) begin
    case (current_command)
      CMD_RESET : $display("Command: RESET");
      CMD_READ : $display("Command: READ");
      CMD_WRITE : $display("Command: WRITE");
      CMD_STATUS : $display("Command: STATUS");
      default : $display("Invalid Command");
    endcase
  end
  
  initial begin
    current_command = CMD_RESET;
    #10 current_command = CMD_READ;
    #10 current_command = CMD_WRITE;
    #10 current_command = CMD_STATUS;

  end
endmodule
```

Here the command values are explicitly set to 2-bit values. This gives you more control and reduces debugging issues down the road.

Now here’s something a lot of people miss you are not stuck with just using enums with registers. Enums are especially useful in parameter declarations. This can dramatically improve the readability of your code. Here's an example where I define different opcodes as enum parameters that can be used to perform some arithmetic operation:

```verilog
module enum_parameter;

 typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV
 } opcode_t;

 parameter  opcode = OP_ADD;
 parameter  A = 10;
 parameter  B = 5;
 reg [31:0] result;
 
 always @(*) begin
   case (opcode)
    OP_ADD: result = A + B;
    OP_SUB: result = A - B;
    OP_MUL: result = A * B;
    OP_DIV: result = A / B;
    default: result = 0;
   endcase
   $display("Opcode: %d result = %d",opcode,result);
 end
  
  initial begin
    #10; 
  end

endmodule
```

With this you can make your designs more configurable and clear when creating modules that do a specific task and when reusing modules as well.

One word of caution always make sure that your reg or wire or whatever you use to store the enum value is big enough to contain all the assigned values. If you're not careful you could be creating a hardware monster that behaves erratically. A real head scratcher for you I bet! If you think about it it is like asking someone to store 10 things inside of a box that can only fit 4 items you will loose some items in the process.

It helps your team also when using names instead of numbers in your logic. One day I was reviewing a legacy design when I saw a state machine with a bunch of numeric state transitions and it made me so angry. I swear I had to create a state diagram on paper to understand each value what it means.

Speaking of making things easier on the team I should say that you might want to put your enum declarations in a separate file which can then be included in all modules that need it. That way you don't have to keep copy pasting the definitions. It keeps your code modular and clean so when you do need to change one you only need to modify that one place. It also helps a lot with collaboration.

And if you do forget to specify the values the synthesizer will assign them automatically so don't worry too much about the numeric values if it is not something that really depends on an existing system.

Now I should say that if you are thinking of something more complex you will want to also check other ways of abstracting state machines. I would recommend looking at "Digital Design Principles and Practices" by John Wakerly it covers state machines and more advanced topics way better than I can explain them here. It's a bit old but the core principles still stand. And if you're more interested in the modern stuff I would suggest also reading some of the SystemVerilog literature out there like "SystemVerilog for Verification" by Chris Spear which will give you a more concrete and modern view of the same principles. It might give you even more tools to manage your states like using structs.

In closing I will say that using enums in Verilog is all about making your code easier to read understand and maintain. It’s not a magic bullet but it’s a really useful tool to have in your toolbox. And believe me you will be happy you learned it especially when you find yourself in the position that I was back in the day. I hope this helped you a little bit and may your code always compile without warnings. You got this.
