---
title: "verilog typedef struct usage?"
date: "2024-12-13"
id: "verilog-typedef-struct-usage"
---

lets dive into this verilog `typedef struct` thing I've seen this pop up a fair bit and I've wrestled with it more times than I'd like to admit so let me share what I've learned its not exactly rocket science but there are definitely some things that can trip you up

 so first off `typedef struct` in verilog is basically your way of creating custom data types you're essentially saying hey compiler instead of using just `reg` or `wire` I want to bundle a bunch of different signals together and give it a name this is super handy for managing complex data structures and makes your code much more readable and maintainable trust me on this one

Think of it like building a mini-container for related signals its a way to group things together logically and avoid having a bunch of loose wires floating around your design for example if you're building a data packet you might have a header a payload and a checksum all those things belong together right `typedef struct` lets you define that relationship

Now lets look at a basic example I'll keep it simple to begin with

```verilog
typedef struct packed {
  logic [7:0] header;
  logic [31:0] payload;
  logic [3:0] checksum;
} packet_t;

module simple_example (
  input logic clk,
  input logic rst,
  input packet_t input_packet,
  output logic [7:0] out_header
);

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      out_header <= 8'h0;
    end else begin
      out_header <= input_packet.header;
    end
  end
endmodule
```

breakdown time `typedef struct packed` this is where you create your struct type `packed` is important it tells verilog to pack these fields together as tightly as possible without any padding between them this is usually what you want when dealing with hardware especially memory mapped areas

Then inside the curly braces you declare your members each member has a name and a data type in this case we have `header` a `payload` and `checksum` after that `packet_t` becomes the name of your custom type you can now declare variables as being of type `packet_t` like in our module where `input_packet` is one such variable

Now when you want to access individual fields inside a `packet_t` variable you use the dot `.` notation we access the `header` by saying `input_packet.header` as you can see

I remember my first encounter with this stuff it was an absolute mess the design was a sprawling spaghetti of unconnected signals and I spent hours trying to figure out what connected to what I wish I had started using `typedef struct` sooner life would have been much simpler

And talking about complex situations let's consider this example a FIFO (First-In First-Out) buffer this is where a struct can become incredibly handy we need to store not just data but also associated control bits like full and empty flags

```verilog
typedef struct packed {
  logic valid;
  logic [31:0] data;
} fifo_element_t;

module fifo_example (
  input logic clk,
  input logic rst,
  input logic push,
  input logic [31:0] push_data,
  input logic pop,
  output logic [31:0] pop_data,
  output logic empty,
  output logic full
);

  localparam DEPTH = 8;
  fifo_element_t fifo_mem [DEPTH - 1:0];
  logic [3:0] head;
  logic [3:0] tail;

  assign empty = (head == tail);
  assign full = ( (head + 1) % DEPTH == tail );

  always_ff @(posedge clk or posedge rst) begin
      if (rst) begin
        head <= 4'h0;
        tail <= 4'h0;
      end else begin
        if (push && !full) begin
          fifo_mem[head].valid <= 1'b1;
          fifo_mem[head].data <= push_data;
          head <= (head + 1) % DEPTH;
        end
          if(pop && !empty) begin
              pop_data <= fifo_mem[tail].data;
              fifo_mem[tail].valid <= 1'b0;
              tail <= (tail + 1) % DEPTH;
        end
      end
  end
endmodule
```

Look at how neat that `fifo_element_t` definition is it bundles the data and a validity flag together making it way easier to handle each entry in our FIFO buffer and in this case the FIFO itself is a memory array of this struct type the validity bit lets us know if there's valid data at a certain index or not super useful for controlling the logic and the data

Now accessing fields like `fifo_mem[head].valid` is very intuitive you see the index into the memory followed by a dot then the name of the field within the struct this is so much cleaner than keeping track of separate arrays for the data and the valid bits it really does reduce the chance of bugs because everything is logically associated

One gotcha that I've seen people fall into even myself sometimes is this notion of the same name being used in different modules or scopes you can’t assume that two `typedef struct` definitions with the same name are compatible across modules if they are not defined in the same scope it might work in some simulation environments or if you are lucky with linting but in general it is bad practice to assume this they are not necessarily the same type so you must be careful about it you don’t want to end up with some weird mismatches

And just when you think you've mastered struct types you realize that you need to do something wild like passing a struct to a function that's where things get a little more involved you'll find that if your code is complex enough you might need to use a function to encapsulate some behavior but you can not directly pass a struct as an argument in verilog or system verilog if they are not of the same defined scope it is not as straightforward as with other languages but you can pack the struct into an array using the `packed` property from the struct definition and pass the array as an argument to the function then you can unpack inside the function again into a struct of the same type

```verilog
typedef struct packed {
  logic [7:0] addr;
  logic [31:0] data;
} mem_access_t;

function automatic logic [39:0] pack_mem_access (input mem_access_t access);
    pack_mem_access = {access.addr,access.data};
endfunction

function automatic mem_access_t unpack_mem_access(input logic [39:0] packed_data);
     mem_access_t access;
     access.addr = packed_data[39:32];
     access.data = packed_data[31:0];
    return access;
endfunction

module function_example (
  input logic clk,
  input logic rst,
  input logic [7:0] addr_in,
    input logic [31:0] data_in,
  output logic [31:0] data_out
);
 mem_access_t my_access;
 logic [39:0] packed_access;

 always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      data_out <= 32'h0;
    end else begin
        my_access.addr <= addr_in;
        my_access.data <= data_in;
        packed_access <= pack_mem_access(my_access);
         data_out <= unpack_mem_access(packed_access).data;

    end
  end
endmodule
```

In this snippet we have the `pack_mem_access` function it takes the `mem_access_t` struct packs it into an array `logic [39:0]` this is important because the size of the array matches the total number of bits in the struct also there is `unpack_mem_access` it unpacks an array into a struct type using the inverse of the pack procedure making sure that the size matches the original struct size

Now you're not strictly limited to simple bit vectors structs can also contain other structs making your data structures even more complex just keep in mind that the overall size of the struct will add up fast if you start making very large nested structs

The most important thing I can tell you is to always strive to make code readable and easy to understand you might think you understand what you wrote at first glance but a few months down the line you'll be cursing your past self if your code is a mess `typedef struct` helps you with this aspect if you use it wisely

As for resources I would really recommend a few papers from some of the more established members of the verilog community there is a very good paper from Clifford Cummings called “SystemVerilog Data Type Strategies” it really explores various types and ways of using them also there is a good book "Digital Design with SystemVerilog" by Brian Holdsworth and Daniel Clasen the key thing is to start with simple examples first then slowly build up your knowledge this is a deep area of knowledge

Remember that the key is practicing and experimentation use it in your projects and slowly you will see the benefits that this brings to your designs also if your code ever starts looking like a labyrinth consider a re-factor maybe you could’ve avoided all of that mess with a good struct from the start now you might feel like you have a PHD in typedef struct I know it feels like that sometimes the important thing is to keep practicing you got this and do not forget that we are all just bunch of registers in a larger hardware simulation called life so dont sweat too much on the small stuff
