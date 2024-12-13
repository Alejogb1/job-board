---
title: "system verilog conditional type definition?"
date: "2024-12-13"
id: "system-verilog-conditional-type-definition"
---

Alright so conditional type definition in SystemVerilog you're asking about that huh Yeah I've been there wrestled with that beast myself more times than I care to remember It's not exactly a walk in the park but it's definitely doable once you get the hang of it lets break it down in a straightforward way

First off when you say conditional type definition you're not really doing like a if else statement that changes the fundamental type itself at compile time like in c++ templates thats not what systemverilog is designed for Instead what we're talking about is using `typedef` to give names to types and then controlling which type name is used based on some condition often through parameters.

Think about it like this imagine you are building a memory controller interface right You might need to have different widths for different system configurations So instead of having to change every single module every time you change the interface bus width you can use this approach to parameterize the data bus.

The most common use case for conditional type definitions I've found boils down to using `typedef` in conjunction with parameterized modules or interfaces

Let me give you an example a very basic one lets say we are building a simple FIFO we can use this technique

```systemverilog
module fifo #(
  parameter DATA_WIDTH = 8
  parameter DEPTH = 16
  parameter USE_WIDE_PTRS = 0
  )
  (input  logic clk
  ,input  logic rst
  ,input  logic [DATA_WIDTH-1:0] data_in
  ,input  logic wr_en
  ,output logic [DATA_WIDTH-1:0] data_out
  ,input  logic rd_en
  ,output logic full
  ,output logic empty
  );


  // Conditional pointer definition
  typedef logic [$clog2(DEPTH)-1:0]  small_ptr_t;
  typedef logic [$clog2(DEPTH+1)-1:0] wide_ptr_t;

  typedef  (USE_WIDE_PTRS == 1) ? wide_ptr_t : small_ptr_t ptr_t;

  ptr_t wr_ptr;
  ptr_t rd_ptr;
  //Rest of the fifo implementation
//fifo memory here etc etc
endmodule

```

Notice the conditional type definition using ternary operator this is what I am talking about this is the crux of what we are talking about the type ptr_t can either be of `small_ptr_t` or `wide_ptr_t` depending on the parameter `USE_WIDE_PTRS`. If USE_WIDE_PTRS is equal to 1 then `ptr_t` will be `wide_ptr_t` otherwise it'll be `small_ptr_t`.

This example shows an often use case where you have two possible types depending on the parameter, in this case an enumerated type can be used to specify a possible number of configurations as well

Lets say you are building an interface with different address bus sizes here an example

```systemverilog
interface mem_if #(
  parameter ADDR_WIDTH_SEL = 0,
    parameter ADDR_WIDTH_32 = 32,
    parameter ADDR_WIDTH_64 = 64
  );

  typedef logic [ADDR_WIDTH_32-1:0] addr_32_t;
  typedef logic [ADDR_WIDTH_64-1:0] addr_64_t;

  typedef enum {ADDR_32, ADDR_64} addr_sel_t;
  localparam addr_sel_t ADDR_SEL = addr_sel_t'(ADDR_WIDTH_SEL);

  typedef (ADDR_SEL == ADDR_32) ? addr_32_t : addr_64_t addr_t;
    logic clk;
    logic reset;
  logic rd_en;
    logic wr_en;
    addr_t addr;
    logic [63:0] data;
  modport master(input clk, input reset, input rd_en,input wr_en, output addr, output data);
modport slave(input clk, input reset, output rd_en, output wr_en,input addr,input data);


endinterface
```

Here you see how with an enum and ternary operator the correct type can be selected based on the parameter provided by the user

The use of localparams is extremely important because parameters are essentially constants and not variables and can't be used in conditions where the condition would need to be evaluated at runtime In these examples because I am using constant parameters they are all evaluated at compile time

Now you might be wondering "Why not just use `if else` or `case` statements directly to choose a data width inside a module?". Well you *could* but it quickly becomes a mess if you have a lot of different places where the type needs to change. With `typedef` you define your type once based on parameters and then use that `typedef` everywhere else in the module or interface. It makes the code easier to read and maintain.

Lets say for some reason you need to use a struct that has different members depending on the type another example can be this

```systemverilog

module type_struct #(
  parameter USE_EXTENDED_DATA = 0
  );

  typedef struct packed {
     logic [7:0] header;
    logic [31:0] data;
  } base_data_t;

typedef struct packed {
     logic [7:0] header;
    logic [31:0] data;
    logic [15:0] crc;
}extended_data_t;


typedef  (USE_EXTENDED_DATA == 1) ? extended_data_t : base_data_t my_data_t;

  my_data_t data_packet;

  initial begin
   data_packet.header = 8'hAA;
    data_packet.data = 32'h12345678;
    if(USE_EXTENDED_DATA == 1)
    data_packet.crc = 16'hABCD;
     $display("Header: %h Data: %h Crc: %h", data_packet.header,data_packet.data, (USE_EXTENDED_DATA == 1) ? data_packet.crc : 16'h0000 );

  end

endmodule
```

Here in the example a struct definition is selected based on a parameter. The way to access a conditional field in the struct is done using a conditional statement outside the definition of the struct type itself. There aren't a way to have members in the struct itself conditionalized with ternary operators. But with this approach we can select different structures based on parameters.

Now a common mistake I've seen people make is trying to use these conditional types *everywhere*. Remember that the conditional type selection has to be resolvable at compile time. You can't use run-time values or signals to decide what the type is going to be. The parameters or localparams need to be constant when evaluating the condition for the typedef. I've seen people try that and they get a lot of confusing errors. Dont do that unless you really know what you are doing you are going to pull your hair off.

About resources well I really found the SystemVerilog standard ieee 1800 a very valuable asset there's really no substitute for that one and you can get it from ieee website. I would recommend reading the part that talks about parameterized types and type definition with `typedef` there a lot of corner cases there. Also a good book is "SystemVerilog for Verification" by Chris Spear I think it will give you a good perspective on these features and what they were designed for its a great book for anyone doing verification also. And one more great book I would recommend is "Writing Testbenches Using SystemVerilog" by Janick Bergeron its a great book if you want to take a deep dive in advanced testbench techniques and how the systemverilog features are used.

One more thing and this is very important to remember I forgot to say this before you have to be careful with type compatibility in systemverilog even if the types are the same size the name of the type matters. There is a concept called type equivalence or structural equivalence. If the type is not the exact type name you have defined it is not type compatible it is another type that happens to have the same number of bits. That might sound like a dumb thing and it is but this is what the standard and the tools say so don't mess around if you don't want the tools yelling at you.

So yeah that's basically it Conditional type definition in SystemVerilog It's a really powerful tool but use it wisely and don't go crazy with it. There are times that it's not even necessary and it can be a headache for someone else reading your code if it is overdone. And remember to read the manual seriously nobody wants to debug someone elses code that is doing black magic with the type definitions without documentation. Oh and a little joke why did the programmer quit his job because he didn't get arrays. Ok ok I will stop now. Good luck coding!
